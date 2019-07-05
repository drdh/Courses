import edu.stanford.nlp.pipeline.*;
import java.util.*;

import org.json.JSONObject;

import base.JsonFile;

public class NLPTest {

    public static void main(String[] args) {

        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        //props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // read some text in the text variable
        //String text = "...";
        JSONObject oj=JsonFile.readFile("D:\\document\\web\\data\\text\\2018-09-1536049348.json");
  		String text=oj.getString("contents");

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

    }

}