// In this we have considered that d can be <=2 like if d=2 then
// d between w1 and w2 can be at max 2 words like d can be equal to 1

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;

public class Pairs_MapClassLevel {

    public static class PairMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Text pair = new Text();

        private Set<String> topWords = new HashSet<>();
        private int distance;

        //Global map (entire mapper)
        private HashMap<String, Integer> globalMap;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            distance = conf.getInt("distance", 1);

            globalMap = new HashMap<>();

            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    Path path = new Path(uri.getPath());
                    loadTopWords(path.getName());
                }
            }
        }

        private void loadTopWords(String filePath) throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            while ((line = reader.readLine()) != null) {
                // assuming format: word count
                String[] parts = line.split("\\s+");
                if (parts.length > 0) {
                    topWords.add(parts[0].toLowerCase().trim());
                }
            }
            reader.close();
        }

        private boolean isValidToken(String token) {
            if (token == null) return false;
            token = token.trim();
            if (token.length() <= 2) return false;
            if (!token.matches("[a-z]+")) return false;
            if (token.matches("^(i|ii|iii|iv|v|vi|vii|viii|ix|x)+$")) return false;
            return true;
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");

            for (int i = 0; i < tokens.length; i++) {

                String w1 = tokens[i];
                if (!isValidToken(w1)) continue;
                if (!topWords.contains(w1)) continue;

                for (int j = i + 1; j <= i + distance && j < tokens.length; j++) {

                    String w2 = tokens[j];
                    if (!isValidToken(w2)) continue;
                    if (!topWords.contains(w2)) continue;

                    String keyPair = w1 + "," + w2;

                    globalMap.put(keyPair, globalMap.getOrDefault(keyPair, 0) + 1);
                }
            }
        }

        // Emit once at end
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            for (String k : globalMap.keySet()) {
                pair.set(k);
                context.write(pair, new IntWritable(globalMap.get(k)));
            }
        }
    }

    public static class PairReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        // args: input output -topwords file -d value
        int d = 1;

        for (int i = 0; i < args.length; i++) {
            if ("-d".equals(args[i])) {
                d = Integer.parseInt(args[++i]);
            }
        }

        conf.setInt("distance", d);

        Job job = Job.getInstance(conf, "CoOccurrence Pairs");

        job.setJarByClass(Pairs_MapClassLevel.class);
        job.setMapperClass(PairMapper.class);
        job.setReducerClass(PairReducer.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); //64 MB

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setNumReduceTasks(1); // important for global counts

        for (int i = 0; i < args.length; i++) {
            if ("-topwords".equals(args[i])) {
                job.addCacheFile(new Path(args[++i]).toUri());
            }
        }

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.out.println("d = " + d);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}