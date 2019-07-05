package base;

import java.io.BufferedReader;  
import java.io.File;  
import java.io.FileReader;  
import java.io.FileWriter;  
import java.io.IOException;  
import java.io.PrintWriter;
import java.util.HashMap;

import org.json.JSONException;  
import org.json.JSONObject;  
  
@SuppressWarnings("unchecked")
public class JsonFile {  
    public static void main(String[] args) throws JSONException, IOException {    
        JSONObject jsonObject = new JSONObject();  
        //jsonObject.put("1", "һ");  
         
        HashMap<String,Integer>map=new HashMap<String,Integer>();
        //map.put("key", 123);
        //map.put("k2",34);
        //jsonObject.put("PERSON",map);
        
        //System.out.println(jsonObject.has("8"));
  
        writeFile("D:\\document\\web\\data\\t.json", jsonObject);  
        
        JSONObject jo=readFile("D:\\document\\web\\data\\t.json");
        System.out.println(jo.toString());
        /*
        System.out.println(jo.get("PERSON"));
        JSONObject jo2=(JSONObject)jo.get("PERSON");
        System.out.print(jo2.get("key"));
        */
        /*
         	JSONObject oj=JsonFile.readFile("D:\\document\\web\\data\\text\\2018-09-1536049348.json");
	  		String text=oj.getString("contents");
         */
        
    }  
  
    public static void writeFile(String filePath, JSONObject JO)  
            //throws IOException{ 
    {
    	try {
    		String sets=JO.toString();
            FileWriter fw = new FileWriter(filePath);  
            PrintWriter out = new PrintWriter(fw);  
            out.write(sets);  
            out.println();  
            fw.close();  
            out.close();     
        } catch (IOException e) {}  
    	
    }  
  
    public static JSONObject readFile(String path) {  
        File file = new File(path);  
        BufferedReader reader = null;  
        String laststr = "";  
        try {  
            reader = new BufferedReader(new FileReader(file));  
            String tempString = null;  
            while ((tempString = reader.readLine()) != null) {  
                laststr = laststr + tempString;  
            }  
            reader.close();  
        } catch (IOException e) {  
            e.printStackTrace();  
        } finally {  
            if (reader != null) {  
                try {  
                    reader.close();  
                } catch (IOException e1) {  
                }  
            }  
        }  
        return new JSONObject(laststr);
    }  
}  