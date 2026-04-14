import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;

public class WordCount2Mod {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		private boolean caseSensitive = false;
		private Set<String> stopwords = new HashSet<String>();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			caseSensitive = conf.getBoolean("wordcount.case.sensitive", false);
			if (conf.getBoolean("wordcount.skip.patterns", false)) {
				URI[] stopwordURIs = Job.getInstance(conf).getCacheFiles();
				for (URI stopwordURI : stopwordURIs) {
					Path stopwordPath = new Path(stopwordURI.getPath());
					String stopwordFileName = stopwordPath.getName().toString();
					parseSkipFile(stopwordFileName);
				}
			}
		}

		private void parseSkipFile(String fileName) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				String pattern = null;
				while ((pattern = reader.readLine()) != null) {
					stopwords.add(pattern.toLowerCase().trim());
				}
				reader.close();
			} catch (IOException ioe) {
				System.err.println(
						"Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
			}
		}

		private boolean isValidToken(String token) {
			if (token == null) return false;
			token = token.trim();
			if (token.length() <= 2) return false;
			if (!token.matches("[a-z]+")) return false;
			if (token.matches("^(i|ii|iii|iv|v|vi|vii|viii|ix|x)+$")) return false;
			if (stopwords.contains(token)) return false;
			return true;
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
			String[] tokens = line.split("[^a-z]+");
			for (String token : tokens) {
				if (!isValidToken(token)) continue;
				word.set(token);
				context.write(word, one);
			}
		}
	}

	public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();
		private Map<String, Integer> wordCountMap = new HashMap<>();

		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			wordCountMap.put(key.toString(), sum);
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			List<Map.Entry<String, Integer>> list = new ArrayList<>(wordCountMap.entrySet());
			list.sort((a, b) -> b.getValue() - a.getValue());

			int count = 0;
			for (Map.Entry<String, Integer> entry : list) {
				if (count >= 50) break;
				context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
				count++;
			}
		}
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "wordcount2");

		job.setMapperClass(TokenizerMapper.class);
		// job.setCombinerClass(IntSumReducer.class); // enable to use 'local aggregation'
		job.setReducerClass(IntSumReducer.class);

		job.setInputFormatClass(CombineTextInputFormat.class);
        CombineTextInputFormat.setMaxInputSplitSize(job, 67108864); //64 MB

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

        job.setNumReduceTasks(1);

		for (int i = 0; i < args.length; ++i) {
			if ("-skippatterns".equals(args[i])) {
				job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
				job.addCacheFile(new Path(args[++i]).toUri());
			} else if ("-casesensitive".equals(args[i])) {
				job.getConfiguration().setBoolean("wordcount.case.sensitive", true);
			}
		}

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}