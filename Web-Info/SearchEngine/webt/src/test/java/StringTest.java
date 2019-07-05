public class StringTest{
	public static void main(String[]args)
	{
		String a="https://research.cs.wisc.edu/dbworld/messages/2018-10/1538439952.html";
		String []b=a.split("/|\\.");
		for(String s:b)
		{
			System.out.println(s);
		}
		System.out.println(b[8]+"-"+b[9]);
		System.out.println("https://research.cs.wisc.edu/dbworld/messages/"+b[8]+"/"+b[9]+".html");
				
		System.out.println("\n\n\n");
		a="25-Aug-2018";
		b=a.split("-");
		for(String s:b)
		{
			System.out.println(s);
		}
		
		int month;
		switch(b[1]) {
		case "Jan":month=1;break;
		case "Feb":month=2;break;
		case "Mar":month=3;break;
		case "Apr":month=4;break;
		case "May":month=5;break;
		case "Jun":month=6;break;
		case "Jul":month=7;break;
		case "Aug":month=8;break;
		case "Sep":month=9;break;
		case "Oct":month=10;break;
		case "Nov":month=11;break;
		case "Dec":month=12;break;
		default:month=0;
		}
		System.out.println(month);
		
	}
}