<%@ page import = "base.Searcher,base.ProcessMainPage,base.EntityMention,base.Indexer,java.util.Map, java.util.ArrayList" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>

<%
    String queryText="input", queryPage, queryNext, queryLast;
    String strHistory;
    String option,dayu,dayl,monthu,monthl,yearu,yearl,date="";
    String update;
    request.setCharacterEncoding("UTF-8");
    queryText = request.getParameter("query");
    queryPage = request.getParameter("page");
    queryLast = request.getParameter("last");
    strHistory = request.getParameter("history");
    option=request.getParameter("option");
    update=request.getParameter("update");
    
    dayu=request.getParameter("dayu");
    monthu=request.getParameter("monthu");
    yearu=request.getParameter("yearu");
    
    dayl=request.getParameter("dayl");
    monthl=request.getParameter("monthl");
    yearl=request.getParameter("yearl");
    
    date=request.getParameter("date");
    if(strHistory == null)
        strHistory = "    ";
    if(strHistory != null && !strHistory.equals("null") && queryText != null && !strHistory.contains(queryText))
        strHistory = "    "+queryText + "<br>" +strHistory ;
    if(queryPage == null || queryPage.equals(""))
        queryPage = "1";
    queryNext = Integer.toString(Integer.parseInt(queryPage));
    if(queryLast != null && !queryLast.equals(queryText)){
        queryPage = "1";
        queryNext = "2";
    }
%>
<html style="margin:3px 50px font-size: small;
    font-family: arial,sans-serif;font-size: small; ">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <title><%= queryText %> - drdh's search engine</title>
</head>

<body >
<div >
<form method = "POST" action = "result.jsp">
    <input type = "text" name = "query" style = "width:300px;height:30px" value="<%= queryText==null?"":queryText %>" placeholder="input text">
    <select name="option" style = "height:30px">
    	<option label="contents" value="contents" >contents</option>
    	<option label="subject" value="subject" >subject</option>
    	<option label="MISC" value="MISC" >MISC</option>
    	<option label="CITY" value="CITY" >related city</option>
    	<option label="COUNTRY" value="COUNTRY" >related country</option>
    	<option label="no" value="no" >nope</option>
    </select>
    <input type = "submit" value = "search" style = "width:80px;height:30px">
    
 
    <input type = "submit" value = "jump to" style = "width:70px;height:30px">
    <input type = "text" name = "page" style = "width:50px;height:30px" value="<%= queryNext %>" >
    <input type = "hidden" name = "history" value="<%= strHistory %>" >
    <input type = "hidden" name = "last" value="<%= queryText %>" >
    

    <br><br>

    <div style="float:left" >
   	<em>upper bound</em>
    <input type="text" name="dayu" style = "width:80px;height:20px" value="<%= dayu==null?"":dayu %>" placeholder="day: 1">
    <input type="text" name="monthu" style = "width:80px;height:20px" value="<%= monthu==null?"":monthu %>" placeholder="month: 10">
    <input type="text" name="yearu" style = "width:80px;height:20px" value="<%= yearu==null?"":yearu %>" placeholder="year: 2018">
    <br><em>lower bound</em>
    <input type="text" name="dayl" style = "width:80px;height:20px" value="<%= dayl==null?"":dayl %>" placeholder="day: 1">
    <input type="text" name="monthl" style = "width:80px;height:20px" value="<%= monthl==null?"":monthl %>" placeholder="month: 10">
    <input type="text" name="yearl" style = "width:80px;height:20px" value="<%= yearl==null?"":yearl %>" placeholder="year: 2018">
    </div>
    <div style="float:left" >
    <div style="float:left">
    <select name="date" style = "width:80px;height:40px">
    	<option label="no" value="no" >nope</option>
    	<option label="deadline" value="deadline" >deadline</option>
    	<option label="sent" value="sent" >sent</option>
    </select>
    </div>
    <div style="float:right">
    <button type = "submit" name="update",value = "update" style = "width:80px;height:40px">update</button>
    </div>
    </div>
    <br><br>
</form>
</div>

<br><br>

<div style="margin:1px 15px;border:3px 50px;padding:1px 50px;width:453px" >
<%
    System.out.println(queryText);
    boolean flag = false;
    //Searcher searcher = new Searcher();
    ArrayList<Map<String,String>> totalResults, results=new ArrayList<>();
    int Page = 1, size = 0;
    
    
    if( queryPage != null && queryPage.length()!=0)
        Page = Integer.parseInt(queryPage);
    if(update!=null){
    	System.out.println("update");
    	out.print("<font color = \"red\" size = \"8\"> update now!<font>");
    	ProcessMainPage.start();
    	out.print("<br><font color = \"green\" size = \"8\">crawl done!<font>");
    	EntityMention.start();
    	out.print("<br><font color = \"green\" size = \"8\">nlp done!<font>");
    	Indexer.start();
    	out.print("<br><font color = \"green\" size = \"8\">update done!<font>");
    }
    else if(queryText != null){
		
    	//String indexDir =  "D:\\document\\web\\data\\index\\";
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
    	String[]q= new String[7];
    	int[]lower7=new int[3];//{2,9,2018};day,month,year
		int[]upper7=new int[3];
		int[]lower8=new int[3];
		int[]upper8=new int[3];
    	int[]mode=new int[9];
    	
    	if(date==null);
    	else if(date.equals("no")){
    		mode[7]=mode[8]=0;
    	}
    	else if(date.equals("sent")){
    		mode[7]=1;
    		lower7[0]=Integer.parseInt(dayl);
    		lower7[1]=Integer.parseInt(monthl);
    		lower7[2]=Integer.parseInt(yearl);
    		
    		upper7[0]=Integer.parseInt(dayu);
    		upper7[1]=Integer.parseInt(monthu);
    		upper7[2]=Integer.parseInt(yearu);
    	}
    	else if(date.equals("deadline")){
    		mode[8]=1;
    		lower8[0]=Integer.parseInt(dayl);
    		lower8[1]=Integer.parseInt(monthl);
    		lower8[2]=Integer.parseInt(yearl);
    		
    		upper8[0]=Integer.parseInt(dayu);
    		upper8[1]=Integer.parseInt(monthu);
    		upper8[2]=Integer.parseInt(yearu);
    	}
    	else{
    		mode[7]=mode[8]=0;
    	}
    	
    	
    	if(option.equals("contents") ){
    		q[0]=queryText;mode[0]=1;
    		//mode[6]=0;
    		System.out.println("contents");
    	}	
    	else if(option.equals("subject")){
    		q[6]=queryText;mode[6]=1;
    		System.out.println("subject");
    	}
    	else if(option.equals("MISC")){
    		q[1]=queryText;mode[1]=1;
    		System.out.println("MISX");
    	}
    	else if(option.equals("CITY")){
    		q[2]=queryText;mode[2]=1;
    		System.out.println("CITY");
    	}
    	else if(option.equals("COUNTRY")){
    		q[3]=queryText;mode[3]=1;
    		System.out.println("COUNTRY");
    	}
    	else if(option.equals("no")){
    		mode[0]=0;
    	}
    	else{
    		q[0]=queryText;mode[0]=1;	
    	}
    				
		totalResults = Searcher.start(q,lower7,upper7,lower8,upper8,mode);
		
        size = totalResults.size();
        out.println("<font color = \"#808080\" size = \"4\">" + size + " results, page");
        out.println(Page +  " of " + (size / 15 + 1) );
        out.println("</font>" + "<br><br>");
        if(size > (Page-1)*15 && Page > 0)
            for (int i = 0; i < 15; i++) {
                if((Page-1)*15+i < size)
                    results.add(totalResults.get((Page-1)*15+i));
            }
        else if(size != 0){
            out.println("<font color = \"red\" size = \"4\">");
            out.print("sorry,out of page range: 1 - " + (size/15+1));
            out.println("</font>" + "<br>");
            flag = true;
        }
        if(results != null && results.size() != 0 ){
            //String strBody, strTitle, strUrl, strScore, strKeywords;
            String path,contents,url,subject,MISC,CITY,COUNTRY,score,webPage,sent,deadline;
            String type,from;
            out.print("<div>");
            for(int i = 0 ; i < results.size() ; i ++){
                Map<String,String> map = results.get(i);
   
                path=map.get("path");
                contents=map.get("contents");
                url=map.get("url");
                subject=map.get("subject");
                score=map.get("score");
                webPage=map.get("webPage");
                MISC=map.get("MISC");
                CITY=map.get("CITY");
                COUNTRY=map.get("COUNTRY");
                sent=map.get("sent");
                deadline=map.get("deadline");
                type=map.get("type");
                from=map.get("from");
                
                
                out.print("<div>");
                	out.print("<div style=\"padding:2px 0px\">");
                		
                		out.print("<font color = \"#1a0dab\" size = \"4\">");
                			out.print("<a href="+url+">");
                			out.print(subject);
                			out.print("</a>");
                		out.print("</font>");
                		
                		out.print("<font color = \"black\" size = \"3\">");
    						out.print("[Score: <strong>"+score+"</strong>]");
    					out.print("</font>");
                	out.print("</div>");
                	out.print("<div style=\"padding:5px 0px\">");
                		
                		
                		out.print("<span ><a href="+webPage+">");
        				out.print("<font color = \"#006621\" size = \"3\">");
        					out.print(webPage);
        				out.print("</font>");
        				out.print("</a></span>");
        				
                		out.print("<span style=\"padding:3px\"><font color = \"#808080\" size = \"3\">");
                		out.print("sent: "+sent+"\tdeadline: "+deadline);
                		out.print("</font><span>");
                	
            			
            		out.print("</div>");
                	
                	out.print("<div style=\"padding:5px 0px\">");
                		out.print("<font color = \"black\" size = \"2\">");
                			out.print(contents);
                		out.print("</font>");
                	out.print("</div>");
                	
                	out.print("</div style=\" ;padding:8px 0px\">");
                		out.print("<font color = \"black\" size = \"2\">");
                		out.print("<strong>MISC</strong>: <em>"+MISC+"</em><br>");
        				out.print("<strong>related city</strong>: <em>"+CITY+"</em><br><strong>related country</strong>: "+COUNTRY+"<br>");
        				out.print("<strong>type</strong>: <em>"+type+"</em><strong>from</strong>: <em>"+from+"</em><br>");
        				out.print("</font>");	
                	out.print("<div>");
                out.print("</div>");
                out.print("<br><br>");
                
				
                
                /*
                out.println("<font color = \"blue\" size = \"4\">");
                out.print("<a href=\"" + url + "\">" + subject + "</a>");
                out.println("</font>" + "<br>");
                out.println("<font color = \"black\" size = \"1\">" + "<strong>score</strong>: " + score + 
                			"          <strong>MISC</strong>:" + (MISC==null?"":MISC));
                out.println("</font>" + "<br>");
                out.println("<font color = \"black\" size = \"3\">" + contents + "<br>" + "<br>");
                */
                
                
            }

            out.print("</div>");
            out.println("<br>");
        }
        else if(flag == false){
            out.println("<font color = \"red\" size = \"4\">");
            out.print("sorry,no result about " + queryText);
            out.println("</font>" + "<br>");
        }
        out.println("History<br>");
        out.println("<font color = \"purple\" size = \"3\">");
        out.println(strHistory);
      	//out.println("<p>"+(new java.util.Date()).toLocaleString() +"</p>");
        out.println("</font>" + "<br>");
    }
    
    
%>
</div>

</body>
</html>