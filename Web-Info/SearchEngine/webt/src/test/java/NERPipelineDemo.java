import edu.stanford.nlp.pipeline.*;

import java.util.Properties;
import java.util.stream.Collectors;

import org.json.JSONObject;

import base.JsonFile;

public class NERPipelineDemo {

  public static void main(String[] args) {
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
    
    //JSONObject oj=JsonFile.readFile("D:\\document\\web\\data\\text\\2018-09-1536049348.json");
	//String text=oj.getString("contents");
    String text="The Program Committee of the 16th European Conference on Logics in Artificial Intelligence (JELIA 2019) invites the submission of technical papers for the conference that will be held in Rende, Italy, from May 8th to May 10th, 2019. The aim of JELIA 2019 is to bring together active researchers interested in all aspects concerning the use of logics in Artificial Intelligence to discuss current research, results, problems, and applications of both theoretical and practical nature. JELIA strives to foster links and facilitate cross-fertilization of ideas among researchers from various disciplines, among researchers from academia and industry, and between theoreticians and practitioners.\r\n" + 
    		"";
    CoreDocument doc = new CoreDocument(text);
    // annotate the document
    pipeline.annotate(doc);
    // view results
    System.out.println("---");
    System.out.println("entities found");
    for (CoreEntityMention em : doc.entityMentions())
      System.out.println("\tdetected entity: \t"+em.text()+"\t"+em.entityType());
    System.out.println("---");
    System.out.println("tokens and ner tags");
    String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect(
        Collectors.joining(" "));
    System.out.println(tokensAndNERTags);
  }

}
