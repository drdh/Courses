package base;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
//import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;
import java.util.Set;

import org.json.JSONObject;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;


/*
(12 classes) 
named (PERSON, LOCATION, ORGANIZATION, MISC), 
numerical (MONEY, NUMBER, ORDINAL, PERCENT),
temporal (DATE, TIME, DURATION, SET) 

(additional)(11 classes)
EMAIL, URL, CITY, STATE_OR_PROVINCE, COUNTRY, NATIONALITY, RELIGION, (job) TITLE, IDEOLOGY, CRIMINAL_CHARGE, CAUSE_OF_DEATH 

total: 23 classes.    
*/
@SuppressWarnings("unchecked")
public class EntityMention {
	static int full=0;
	static int empty=0;
  public static HashMap<String,HashMap<String,Integer>>
  entityMention(String path) {
    // set up pipeline properties
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
    // example customizations (these are commented out but you can uncomment them to see the results

    // disable fine grained ner
    //props.setProperty("ner.applyFineGrained", "false");

    // customize fine grained ner
    //props.setProperty("ner.fine.regexner.mapping", "example.rules");
    //props.setProperty("ner.fine.regexner.ignorecase", "true");

    // add additional rules
    //props.setProperty("ner.additional.regexner.mapping", "example.rules");
    //props.setProperty("ner.additional.regexner.ignorecase", "true");

    // add 2 additional rules files ; set the first one to be case-insensitive
    //props.setProperty("ner.additional.regexner.mapping", "ignorecase=true,example_one.rules;example_two.rules");

    // set up pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    // make an example document
    
    JSONObject oj=JsonFile.readFile(path);
    
	String text=oj.getString("contents");
	//System.out.println(text);
	if(text.length()<=5) {
		System.out.println("empty text");
		empty++;
		return new HashMap<String,HashMap<String,Integer>>();
	}
		
	System.out.println("full text");
	full++;
	System.out.println("full: "+full+"\tempty: "+empty+"\ttotal: "+(full+empty));
	//System.out.println(text);
    //String text="The Program Committee of the 16th European Conference on Logics in Artificial Intelligence (JELIA 2019) invites the submission of technical papers for the conference that will be held in Rende, Italy, from May 8th to May 10th, 2019. The aim of JELIA 2019 is to bring together active researchers interested in all aspects concerning the use of logics in Artificial Intelligence to discuss current research, results, problems, and applications of both theoretical and practical nature. JELIA strives to foster links and facilitate cross-fertilization of ideas among researchers from various disciplines, among researchers from academia and industry, and between theoreticians and practitioners.\r\n" ;
    CoreDocument doc = new CoreDocument(text);
    // annotate the document
    pipeline.annotate(doc);
    // view results
    /*
    System.out.println("---");
    System.out.println("entities found");
    for (CoreEntityMention em : doc.entityMentions())
      System.out.println("\tdetected entity: \t"+em.text()+"\t"+em.entityType());
    System.out.println("---");
    System.out.println("tokens and ner tags");
    String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect(
        Collectors.joining(" "));
    System.out.println(tokensAndNERTags);
    */
    //results stats
    HashMap<String,HashMap<String,Integer>> entityMap=new HashMap<String,HashMap<String,Integer>>();
    for (CoreEntityMention em : doc.entityMentions())
    {
    	String enType=em.entityType();
    	String enText=em.text();
    	
    	HashMap<String,Integer> typeMap;
    	if(entityMap.containsKey(enType))
    	{
    		typeMap=entityMap.get(enType);
    		if(typeMap.containsKey(enText))
    		{
    			Integer tmp=typeMap.get(enText);
    			typeMap.put(enText, tmp+1);
    		}
    		else
    		{
    			typeMap.put(enText, 1);
    		}
    	}
    	else 
    	{
    		typeMap=new HashMap<String,Integer>();
    		typeMap.put(enText, 1);
    		entityMap.put(enType, typeMap);
    	}
    }
    
    return entityMap;
  }
  
  public static void printEntity(HashMap<String,HashMap<String,Integer>> entityMap) {
	    //System.out.println(entityMap);
	    Set<HashMap.Entry<String,HashMap<String,Integer>>> es=entityMap.entrySet();
	    Iterator<HashMap.Entry<String,HashMap<String,Integer>>>it=es.iterator();
	    //for(Iterator<HashMap.Entry<String,HashMap<String,Integer>>>it=entityMap.entrySet();it.hasNext();)
	    while(it.hasNext())
	    {
	    	HashMap.Entry<String,HashMap<String,Integer>> en=it.next();
	    	String key=en.getKey();
	    	HashMap<String,Integer>value=en.getValue();
	    	System.out.println(key);
	    	System.out.println(value.size());
	    	System.out.println(value);
	    	
	    } 
  }
  
//mainPage.data 表示当前已爬取的url/filename
//mainPage2.data 需要使用entityMention处理的file
//mainPage3.data 已经被entityMention处理完但还没有建立索引的file
  public static void start() {
	  Set<String> set2;//for entityMention
	  Set<String> set3;//for entityMention
	  //读取
	  try {
			FileInputStream fis2 = new FileInputStream("D:\\document\\web\\data\\record\\mainPage2.data");
			ObjectInputStream ois2 = new ObjectInputStream(fis2);
			set2=(HashSet<String>)ois2.readObject();
			ois2.close();
  	  }catch (Exception e){
			set2=new HashSet<String>();
  	  }
	  
	  try {
			FileInputStream fis3 = new FileInputStream("D:\\document\\web\\data\\record\\mainPage3.data");
			ObjectInputStream ois3 = new ObjectInputStream(fis3);
			set3=(HashSet<String>)ois3.readObject();
			ois3.close();
	  }catch (Exception e){
			set3=new HashSet<String>();
	  }
	  
	  int stop=0;
	  Set<String> setTmp=new HashSet<String>();
	  for(String name:set2) {
		  String path="D:\\document\\web\\data\\text\\"+name+".json";
		  
		  HashMap<String,HashMap<String,Integer>> entityMap=entityMention(path);
		  //printEntity(entityMap);
		  
		  
		  String pathEntity="D:\\document\\web\\data\\entityMention\\"+name+".json";
		  JSONObject jo=new JSONObject(); 
		  
		  //System.out.println(stop);
		  for(HashMap.Entry<String,HashMap<String,Integer>>entry : entityMap.entrySet()) {
		    	String key=entry.getKey();
		    	HashMap<String,Integer>value=entry.getValue();
		    	jo.put(key, value);
		   }
		  JsonFile.writeFile(pathEntity, jo);
		  
		  setTmp.add(name);
		  
		  stop++;
		  //System.out.println(stop);
		  System.out.println(name);
		  if(stop>1000)//以100为一个周期，避免跑到很大数字中途失败，后期的少量文件可以去掉
			  break;
	  }
	  set2.removeAll(setTmp);
	  set3.addAll(setTmp);
	  System.out.println("size: "+set2.size()+"\t"+set3.size());
	  
	  
	  //写入各自的文件
	  try {
			FileOutputStream fos2 = new FileOutputStream("D:\\document\\web\\data\\record\\mainPage2.data");
			ObjectOutputStream oos2 = new ObjectOutputStream(fos2);
			
			oos2.writeObject(set2);
			oos2.flush();
			oos2.close();
			
			//for entityMention
			FileOutputStream fos3 = new FileOutputStream("D:\\document\\web\\data\\record\\mainPage3.data");
			ObjectOutputStream oos3 = new ObjectOutputStream(fos3);
			
			oos3.writeObject(set3);
			oos3.flush();
			oos3.close();
			
		}catch (IOException e)
		{
			e.printStackTrace();
		}
	  
	  
	  
  }

  
  public static void main(String[]args)
  {
	  /*
	  String path="D:\\document\\web\\data\\text\\2018-09-1536049348.json";
	  //printEntity(entityMention(path));
	  entityMention(path);
	  String path2="D:\\document\\web\\data\\text\\2018-09-1536049348.json";
	  //printEntity(entityMention(path2));
	  entityMention(path2);
	  */
	  start();
  }

}
