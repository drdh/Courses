import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.util.Collection;
import java.util.Properties;

import org.json.JSONObject;

import base.JsonFile;

/** A demo illustrating how to call the OpenIE system programmatically.
 */
public class OpenIEDemo {

  public static void main(String[] args) throws Exception {
    // Create the Stanford CoreNLP pipeline
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
    
    //props.setProperty("openie.splitter.disable", "true");
    
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    
    

    JSONObject oj=JsonFile.readFile("D:\\document\\web\\data\\text\\2018-09-1536049348.json");
	String text=oj.getString("contents");
    
    //String text="The Program Committee of the 16th European Conference on Logics in Artificial Intelligence (JELIA 2019) invites the submission of technical papers for the conference that will be held in Rende, Italy, from May 8th to May 10th, 2019. The aim of JELIA 2019 is to bring together active researchers interested in all aspects concerning the use of logics in Artificial Intelligence to discuss current research, results, problems, and applications of both theoretical and practical nature. JELIA strives to foster links and facilitate cross-fertilization of ideas among researchers from various disciplines, among researchers from academia and industry, and between theoreticians and practitioners.\r\n";
    // Annotate an example document.
    Annotation doc = new Annotation(text);
    pipeline.annotate(doc);

    // Loop over sentences in the document
    for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
      // Get the OpenIE triples for the sentence
      Collection<RelationTriple> triples =
	          sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
      // Print the triples
      for (RelationTriple triple : triples) {
        System.out.println(triple.confidence + "\t" +
            triple.subjectLemmaGloss() + "\t" +
            triple.relationLemmaGloss() + "\t" +
            triple.objectLemmaGloss());
      }
    }
  }
}