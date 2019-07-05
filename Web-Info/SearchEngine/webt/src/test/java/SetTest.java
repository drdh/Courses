import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashSet;
import java.util.Set;

@SuppressWarnings("unchecked")
public class SetTest{
	public static void main(String[]args)
	{
		String a="123";
		String b="456";
		String c="123";
		
		Set<String> set=new HashSet<String>();
		set.add(a);
		set.add(b);
		set.add(c);
		
		for(String n:set)
		{
			System.out.println(n);
		}
		
		try {
			FileOutputStream fos = new FileOutputStream("D:\\document\\web\\data\\t.data");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			
			oos.writeObject(set);
			oos.flush();
			oos.close();
			
		}catch (IOException e)
		{
			e.printStackTrace();
		}
	
		System.out.println("After file");
		
		Set<String> set2=new HashSet<String>();
		try {
			FileInputStream fis = new FileInputStream("D:\\document\\web\\data\\t.data");
			ObjectInputStream ois = new ObjectInputStream(fis);
			set2=(HashSet<String>)ois.readObject();
			ois.close();
		}catch (Exception e)
		{
			
		}
		
		String d="123";
		System.out.println(set2.contains(d));
		
		for(String n:set2)
			System.out.println(n);
		
	}
}