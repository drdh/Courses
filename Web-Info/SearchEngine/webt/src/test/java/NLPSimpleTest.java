import java.util.List;

import org.json.JSONObject;

import base.JsonFile;
import edu.stanford.nlp.simple.*;

public class NLPSimpleTest{
	public static void main(String []args)
	{
		//JSONObject oj=JsonFile.readFile("D:\\document\\web\\data\\text\\2018-09-1536049348.json");
  		//String text=oj.getString("contents");
		String text="List<CoreEntityMention> entityMentions = sentence.entityMentions();\r\n" + 
				"    System.out.println(\"Example: entity mentions\");\r\n" + 
				"    System.out.println(entityMentions);\r\n" + 
				"    System.out.println();";
  		
  		Sentence sent = new Sentence(text);
  		List<String> words=sent.words();
  		List<String> nerTags = sent.nerTags();
  		System.out.println(words);
  		System.out.println(nerTags);
  		
  		/*
  		for(String s:nerTags)
  		{
  			System.out.println(s);
  		}
  		*/
	}
}