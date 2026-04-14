//Creating Co-Occurrence matrix using Stripes approach
//Using top50 words only just like in 1b
//Considered all words in the window of size d as co-occurring with w1

import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;

public class CoOccurrenceStripes {

    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int distance;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("distance", 1);

            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    Path path = new Path(uri.getPath());
                    loadTopWords(path.getName());
                }
            }
        }

        private void loadTopWords(String file) throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                topWords.add(line.split("\\s+")[0].toLowerCase().trim());
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
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^a-z]+");

            for (int i = 0; i < tokens.length; i++) {

                String w1 = tokens[i];
                if (!isValidToken(w1)) continue;
                if (!topWords.contains(w1)) continue;

                MapWritable stripe = new MapWritable();

                for (int j = i + 1; j <= i + distance && j < tokens.length; j++) {

                    String w2 = tokens[j];
                    if (!isValidToken(w2)) continue;
                    if (!topWords.contains(w2)) continue;

                    Text neighbor = new Text(w2);

                    if (stripe.containsKey(neighbor)) {
                        IntWritable count = (IntWritable) stripe.get(neighbor);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
                    }
                }

                context.write(new Text(w1), stripe);
            }
        }
    }

    public static class StripeReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            MapWritable result = new MapWritable();

            for (MapWritable stripe : values) {

                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {

                    Text neighbor = (Text) entry.getKey();
                    IntWritable count = (IntWritable) entry.getValue();

                    if (result.containsKey(neighbor)) {
                        IntWritable current = (IntWritable) result.get(neighbor);
                        current.set(current.get() + count.get());
                    } else {
                        result.put(neighbor, new IntWritable(count.get()));
                    }
                }
            }

            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        int d =1;

        for (int i = 0; i < args.length; i++) {
            if ("-d".equals(args[i])) {
                d = Integer.parseInt(args[++i]);
            }
        }

        conf.setInt("distance", d);

        Job job = Job.getInstance(conf, "CoOccurrence Stripes");

        job.setJarByClass(CoOccurrenceStripes.class);
        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(StripeReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);

        job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); //64 MB

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(MapWritable.class);

        job.setNumReduceTasks(1);

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