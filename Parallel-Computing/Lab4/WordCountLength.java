import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class WordCountLength {

  //mapper操作
  public static class TokenizerMapper
       extends Mapper<Object, Text, IntWritable, IntWritable>{
         //           输入key，输入value，输出key，输出value

    private final static IntWritable one = new IntWritable(1);
    private IntWritable intValue=new IntWritable();

    //private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());//得到键值对<word,1>==> <4,1>
      while (itr.hasMoreTokens()) {//是否还有分割符
        //word.set(itr.nextToken());//返回从当前位置到下一个分隔符的字符串
        intValue.set(itr.nextToken().length());
        context.write(intValue, one);//键值对，放在context里
      }
    }
  }

  public static class IntSumReducer
      //reduce 方法的目的就是对列表的值进行加和处理
       extends Reducer<IntWritable,IntWritable,IntWritable,IntWritable> {
         //输入key:单个单词，value:单词的计数值的列表
         //输出的是< key,value>,key 指单个单词，value 指对应单词的计数值的列表的值的总和
    private IntWritable result = new IntWritable();

    public void reduce(IntWritable key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();//默认情况下，Configuration开始实例化的时候，会从Hadoop的配置文件里读取参数。 
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();//从命令行参数里读取参数
    if (otherArgs.length != 2) {
      System.err.println("Usage: wordcount <in> <out>");
      System.exit(2);
    }//在MapReduce处理过程中，由Job对象负责管理和运行一个计算任务，然后通过Job的若干方法来对任务的参数进行设置。”word count”是Job的名字
    Job job = new Job(conf, "word count");
    job.setJarByClass(WordCountLength.class);//根据WordCount类的位置设置Jar文件
    job.setMapperClass(TokenizerMapper.class);//设置Mapper 
    job.setCombinerClass(IntSumReducer.class);//设置Combiner,这里先使用Reduce类来进行Mapper 的中间结果的合并，能够减轻网络传输的压力。 
    job.setReducerClass(IntSumReducer.class);//设置Reduce
    //此处应当需要修改
    job.setOutputKeyClass(IntWritable.class);//设置输出键的类型
    job.setOutputValueClass(IntWritable.class);//设置输出值的类型 
    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));//设置输入文件，它是otherArgs第一个参数 
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));//设置输出文件，将输出结果写入这个文件里，它是otherArgs第二个参数 
    System.exit(job.waitForCompletion(true) ? 0 : 1);//job执行，等待执行结果
  }
}

