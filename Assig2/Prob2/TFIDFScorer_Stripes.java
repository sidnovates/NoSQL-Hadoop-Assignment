package partb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;

import opennlp.tools.stemmer.PorterStemmer;

/**
 * Problem 2b: TF-IDF scoring over Wikipedia articles.
 *
 * Uses a STRIPES algorithm:
 *   Mapper  — accumulates per-document term-count stripes (only for the top-100
 *             terms identified in Problem 2a), then emits one stripe per document
 *             in cleanup() to minimise intermediate shuffle volume.
 *   Reducer — merges all partial stripes for the same document (needed when a
 *             large article is split across multiple mapper tasks), computes
 *             SCORE = TF * log(10000 / DF + 1), and emits one row per
 *             (document, term) pair.
 *
 * Outputs a TSV file with schema:  ID<tab>TERM<tab>SCORE
 *
 * Usage:
 *   hadoop jar problem2.jar partb.TFIDFScorer \
 *     <input-dir>  <output-dir>  <hdfs-path-to-df_top100.tsv>
 */
public class TFIDFScorer_Stripes extends Configured implements Tool {

    private static final String PAIR_SEP = "|";
    private static final String KV_SEP   = ":";

    public static class StripeMapper extends Mapper<Object, Text, Text, Text> {

        // term → DF  (loaded from df_top100.tsv; only these terms are counted)
        private final Map<String, Integer> dfMap = new HashMap<>();

        // docId → { term → count }  (accumulated across all lines of each doc)
        private final Map<String, Map<String, Integer>> docStripes = new HashMap<>();

        private final PorterStemmer stemmer = new PorterStemmer();
        private final Text outKey   = new Text();
        private final Text outValue = new Text();

        /**
         * Runs once per mapper task.
         * Loads df_top100.tsv from the distributed cache into dfMap.
         */
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            URI[] cacheFiles = Job.getInstance(conf).getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) return;

            for (URI uri : cacheFiles) {
                Path filePath = new Path(uri.getPath());
                String fileName = filePath.getName();
                try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\t");
                        if (parts.length == 2) {
                            try {
                                dfMap.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                            } catch (NumberFormatException ignored) { }
                        }
                    }
                }
            }
        }

        /**
         * Called once per input line.
         *
         * Stems each token and, if it is in the top-100 vocabulary, increments
         * its count in the in-memory stripe for this document.
         * Nothing is emitted here — all emits happen in cleanup().
         */
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            String inputFile = conf.get("mapreduce.map.input.file");
            if (inputFile == null) {
                inputFile = conf.get("map.input.file");
            }

            Path path;
            if (inputFile != null) {
                path = new Path(inputFile);
            } else {
                org.apache.hadoop.mapreduce.InputSplit split = context.getInputSplit();
                if (!(split instanceof FileSplit)) {
                    throw new IOException("Unsupported split type: " + split.getClass());
                }
                path = ((FileSplit) split).getPath();
            }

            String filename = path.getName();
            String docId    = filename.replaceAll("\\.[^.]+$", "");

            // Get (or create) the stripe for this document
            Map<String, Integer> stripe = docStripes.computeIfAbsent(docId, k -> new HashMap<>());

            String line    = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");

            for (String token : tokens) {
                if (token.length() < 2) continue;
                String stemmed = stemmer.stem(token);
                if (dfMap.containsKey(stemmed)) {
                    stripe.merge(stemmed, 1, Integer::sum);
                }
            }
        }

        /**
         * Called once per mapper task after all map() calls.
         *
         * Emits (docId, stripe_string) for every document accumulated.
         * Stripe string format: "term1:count1|term2:count2|..."
         * Using pipe as separator because stemmed terms are purely alphabetic.
         */
        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Map<String, Integer>> docEntry : docStripes.entrySet()) {
                outKey.set(docEntry.getKey());

                StringBuilder sb = new StringBuilder();
                for (Map.Entry<String, Integer> termEntry : docEntry.getValue().entrySet()) {
                    if (sb.length() > 0) sb.append(PAIR_SEP);
                    sb.append(termEntry.getKey()).append(KV_SEP).append(termEntry.getValue());
                }
                outValue.set(sb.toString());
                context.write(outKey, outValue);
            }
        }
    }

   public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {

        // Same top-100 DF map, loaded by each reducer task
        private final Map<String, Integer> dfMap = new HashMap<>();
        private final Text outValue = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            URI[] cacheFiles = Job.getInstance(conf).getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) return;

            for (URI uri : cacheFiles) {
                Path filePath = new Path(uri.getPath());
                String fileName = filePath.getName();
                try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\t");
                        if (parts.length == 2) {
                            try {
                                dfMap.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                            } catch (NumberFormatException ignored) { }
                        }
                    }
                }
            }
        }

        /**
         * Receives all partial stripes for one document.
         *
         * Step 1 — Merge stripes: sum counts for the same term across all
         *          received stripe values. This handles the case where a large
         *          article was split across multiple mapper tasks.
         *
         * Step 2 — Score each term:
         *          SCORE = TF * log(10000 / DF + 1)
         *          where TF  = total term count in this document
         *                DF  = number of documents containing this term (from 2a)
         *
         * Emits (docId, "term\tscore") — TextOutputFormat writes "docId\tterm\tscore",
         * matching the required schema:  ID<tab>TERM<tab>SCORE
         */
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            // Merge all partial stripes for this document
            Map<String, Integer> termCounts = new HashMap<>();
            for (Text stripeText : values) {
                String[] pairs = stripeText.toString().split("\\" + PAIR_SEP);
                for (String pair : pairs) {
                    int sep = pair.indexOf(KV_SEP);
                    if (sep < 0) continue;
                    String term = pair.substring(0, sep);
                    try {
                        int count = Integer.parseInt(pair.substring(sep + 1));
                        termCounts.merge(term, count, Integer::sum);
                    } catch (NumberFormatException ignored) { }
                }
            }

            // Compute and emit TF-IDF score for each term
            for (Map.Entry<String, Integer> entry : termCounts.entrySet()) {
                String term = entry.getKey();
                int    tf   = entry.getValue();
                int    df   = dfMap.getOrDefault(term, 1);

                double score = tf * Math.log(10000.0 / df + 1.0);

                // key = docId, value = "term\tscore"
                // TextOutputFormat writes: docId\tterm\tscore  →  ID<tab>TERM<tab>SCORE ✓
                outValue.set(term + "\t" + String.format("%.6f", score));
                context.write(key, outValue);
            }
        }
    }


    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                "Usage: hadoop jar problem2.jar partb.TFIDFScorer " +
                "<input-dir> <output-dir> <hdfs-df_top100-path>");
            return 1;
        }

        Configuration conf = getConf();

        Job job = Job.getInstance(conf, "tfidf_scorer");
        job.setJarByClass(TFIDFScorer_Stripes.class);

        // Add df_top100.tsv to distributed cache (localised to each task node)
        job.addCacheFile(new Path(args[2]).toUri());

        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(TFIDFReducer.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); //64 MB
        

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }


    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new TFIDFScorer_Stripes(), args));
    }
}
