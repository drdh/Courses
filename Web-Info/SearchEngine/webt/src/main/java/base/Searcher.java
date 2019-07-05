package base;

import java.io.StringReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.highlight.Fragmenter;
import org.apache.lucene.search.highlight.Highlighter;
import org.apache.lucene.search.highlight.QueryScorer;
import org.apache.lucene.search.highlight.SimpleHTMLFormatter;
import org.apache.lucene.search.highlight.SimpleSpanFragmenter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/*
 * 0:contents
 * 1:MISC
 * 2:CITY
 * 3:COUNTRY
 * 4:type
 * 5:from
 * 6:subject
 * 7:sentD
 * 8:deadlineD 
 */
public class Searcher {
	//将mode制成可拓展的模式
    public static ArrayList<Map<String,String>> search(String indexDir, String[]q,
    		int[]lower7,int[]upper7,int[]lower8,int[]upper8,int[] mode) throws Exception {

        Directory dir = FSDirectory.open(Paths.get(indexDir)); //获取要查询的路径，也就是索引所在的位置
        IndexReader reader = DirectoryReader.open(dir);
        IndexSearcher searcher = new IndexSearcher(reader);
         
        BooleanQuery.Builder booleanQuery = new BooleanQuery.Builder();
        //查询表达式
        Analyzer analyzer = new StandardAnalyzer(); //标准分词器，会自动去掉空格啊，is a the等单词
        
        if(mode[0]==1) {
        	QueryParser parser0 = new QueryParser("contents", analyzer); //查询解析器
            Query query0 = parser0.parse(q[0]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query0,BooleanClause.Occur.MUST);
        }
        
        if(mode[1]==1) {
        	QueryParser parser1 = new QueryParser("MISC", analyzer); //查询解析器
            Query query1 = parser1.parse(q[1]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query1,BooleanClause.Occur.MUST);
        }
        
        if(mode[2]==1) {
        	QueryParser parser2 = new QueryParser("CITY", analyzer); //查询解析器
            Query query2 = parser2.parse(q[2]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query2,BooleanClause.Occur.MUST);
        }
        
        if(mode[3]==1) {
        	QueryParser parser3 = new QueryParser("COUNTRY", analyzer); //查询解析器
            Query query3 = parser3.parse(q[3]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query3,BooleanClause.Occur.MUST);
        }
        
        if(mode[4]==1) {
        	QueryParser parser4 = new QueryParser("type", analyzer); //查询解析器
            Query query4 = parser4.parse(q[4]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query4,BooleanClause.Occur.MUST);
        }
        
        if(mode[5]==1) {
        	QueryParser parser5 = new QueryParser("from", analyzer); //查询解析器
            Query query5 = parser5.parse(q[5]); //通过解析要查询的String，获取查询对象 
            booleanQuery.add(query5,BooleanClause.Occur.MUST);
        }
        
        if(mode[6]==1) {
        	QueryParser parser6 = new QueryParser("subject", analyzer); //查询解析器
            Query query6 = parser6.parse(q[6]); //通过解析要查询的String，获取查询对象
            booleanQuery.add(query6,BooleanClause.Occur.MUST);
        }
        //数字
        //int[]lower7= {2,9,2018};
        //int[]upper7= {2,9,2018};
        if(mode[7]==1) {
        	Query query7=IntPoint.newRangeQuery("sentD",lower7,upper7);
        	booleanQuery.add(query7,BooleanClause.Occur.MUST);
        }
        
        if(mode[8]==1) {
        	Query query8=IntPoint.newRangeQuery("deadlineD",lower8,upper8);
        	booleanQuery.add(query8,BooleanClause.Occur.MUST);
        }
        
     
        //特定项
        //Term term=new Term("MISC","ACM");
        //query = new TermQuery(term);
       
        long startTime = System.currentTimeMillis(); //记录索引开始时间
        TopDocs docs = searcher.search(booleanQuery.build(),1000);//开始查询，查询前10条数据，将记录保存在docs中
        
        
        long endTime = System.currentTimeMillis(); //记录索引结束时间
        System.out.println("匹配" + q.toString() + "共耗时" + (endTime-startTime) + "毫秒");
        System.out.println("查询到" + docs.totalHits + "条记录");

        SimpleHTMLFormatter simpleHTMLFormatter = new SimpleHTMLFormatter("<b><font color=red>","</font></b>"); //如果不指定参数的话，默认是加粗，即<b><b/>
        QueryScorer scorer = new QueryScorer(booleanQuery.build());//计算得分，会初始化一个查询结果最高的得分
        Fragmenter fragmenter = new SimpleSpanFragmenter(scorer); //根据这个得分计算出一个片段
        Highlighter highlighter = new Highlighter(simpleHTMLFormatter, scorer);
        highlighter.setTextFragmenter(fragmenter); //设置一下要显示的片段

        ArrayList<Map<String, String>> result = new ArrayList<>();
       
        for(ScoreDoc scoreDoc : docs.scoreDocs) { //取出每条查询结果
            Document doc = searcher.doc(scoreDoc.doc); //scoreDoc.doc相当于docID,根据这个docID来获取文档
            //System.out.println(doc.get("fullPath")); //fullPath是刚刚建立索引的时候我们定义的一个字段
            //System.out.println(scoreDoc.score);
            
            String contents=doc.get("contents");
            String contentsH="";
            
            String subject=doc.get("subject");
            String subjectH="";
            
            String MISC=doc.get("MISC");
            String MISCH="";
            
            String CITY=doc.get("CITY");
            String CITYH="";
            
            String COUNTRY=doc.get("COUNTRY");
            String COUNTRYH="";

            

            /*
             * 0:contents
             * 1:MISC
             * 2:CITY
             * 3:COUNTRY
             * 4:type
             * 5:from
             * 6:subject
             * 7:sentD
             * 8:deadlineD 
             */
            
            Map<String,String> map = new HashMap<>();
            map.put("type", doc.get("type"));
            map.put("from", doc.get("from"));
            map.put("sent", doc.get("sent"));
            map.put("subject", doc.get("subject"));
            map.put("url", doc.get("url"));
            map.put("deadline", doc.get("deadline"));
            map.put("webPage", doc.get("webPage"));            
            map.put("path", doc.get("fullPath"));
            map.put("contents", contents);
            map.put("MISC", doc.get("MISC"));
            map.put("CITY", doc.get("CITY"));
            map.put("COUNTRY", doc.get("COUNTRY"));
            map.put("score", String.valueOf(scoreDoc.score));
          //显示高亮
            if(contents!= null) {
                TokenStream tokenStream4contents = analyzer.tokenStream("contents", new StringReader(contents));
                contentsH = highlighter.getBestFragment(tokenStream4contents, contents);
                //System.out.println(contentsH);   
                map.put("contents", contentsH);
            }
            if(mode[6]==1&&subject!=null)
            {
            	TokenStream tokenStream4subject = analyzer.tokenStream("subject", new StringReader(subject));
                subjectH = highlighter.getBestFragment(tokenStream4subject, subject);
                //System.out.println(contentsH);   
                map.put("subject", subjectH);
            }
            if(mode[1]==1&&MISC!=null)
            {
            	TokenStream tokenStream4MISC = analyzer.tokenStream("MISC", new StringReader(MISC));
            	MISCH = highlighter.getBestFragment(tokenStream4MISC, MISC);
                //System.out.println(contentsH);   
                map.put("MISC", MISCH);
            }
            if(mode[2]==1&&CITY!=null)
            {
            	TokenStream tokenStream4CITY = analyzer.tokenStream("CITY", new StringReader(CITY));
            	CITYH = highlighter.getBestFragment(tokenStream4CITY, CITY);
                //System.out.println(contentsH);   
                map.put("CITY", CITYH);
            }
            if(mode[3]==1&&COUNTRY!=null)
            {
            	TokenStream tokenStream4COUNTRY = analyzer.tokenStream("COUNTRY", new StringReader(COUNTRY));
            	COUNTRYH = highlighter.getBestFragment(tokenStream4COUNTRY, COUNTRY);
                //System.out.println(contentsH);   
                map.put("COUNTRY", COUNTRYH);
            }
            
            
            //设置返回值
            
            /*
            System.out.println("type: "+doc.get("type"));
            System.out.println("from: "+doc.get("from"));
            System.out.println("subject: "+doc.get("subject"));
            System.out.println("url: "+doc.get("url"));
            System.out.println("deadline: "+doc.get("deadline"));
            System.out.println("webPage: "+doc.get("webPage"));
            System.out.println("fullPath: "+doc.get("fullPath"));
            System.out.println("MISC: "+doc.get("MISC"));
            System.out.println("CITY: "+doc.get("CITY"));
            System.out.println("COUNTRY: "+doc.get("COUNTRY"));
            System.out.println("end\n\n\n");
            */
            result.add(map);
        }
        reader.close();
        return result;
    }

    public static ArrayList<Map<String,String>> start(String[]q,
    		int[]lower8,int[]upper8,int[]lower9,int[]upper9,int[] mode) {
        String indexDir =  "D:\\document\\web\\data\\index\\";
        //String q = "Conceptual Structures"; //查询这个字符串
        ArrayList<Map<String,String>> result=null;
        try {
            result=search(indexDir,q,
            	lower8,upper8,lower9,upper9,mode);
        } catch (Exception e) {
            e.printStackTrace();
            //result=null;
        }
        return result;
    }
    
    /*
     * 0:contents
     * 1:MISC
     * 2:CITY
     * 3:COUNTRY
     * 4:type
     * 5:from
     * 6:subject
     * 7:sentD
     * 8:deadlineD 
     */
    public static void main(String[] args) {
    	String[]q= new String[7];
    	int[]mode=new int[9];
    	
    	//q[0]="Conceptual Structures";mode[0]=1;
    	//q[1]="ICCS";mode[1]=1;
    	//q[2]="Beijing";mode[2]=1;
    	q[3]="China";mode[3]=1;
    	
    	
		int[]lower7=new int[3];//{2,9,2018};day,month,year
		int[]upper7=new int[3];
		int[]lower8=new int[3];
		int[]upper8=new int[3];
		
    	start(q,lower7,upper7,lower8,upper8,mode);
    }
}

