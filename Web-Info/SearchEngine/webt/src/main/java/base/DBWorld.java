package base;
import java.util.List;
import java.io.*;

import us.codecraft.webmagic.Page;
//import us.codecraft.webmagic.ResultItems;
import us.codecraft.webmagic.Site;
import us.codecraft.webmagic.Spider;
//import us.codecraft.webmagic.Task;
//import us.codecraft.webmagic.pipeline.Pipeline;
//import us.codecraft.webmagic.pipeline.JsonFilePipeline;
import us.codecraft.webmagic.processor.PageProcessor;





public class DBWorld implements PageProcessor {

    private Site site = Site.me().setRetryTimes(10).setSleepTime(1000).setCharset("utf-8");
    
    private int fileId=0;
    private  FileWriter out;

    @Override
    public void process(Page page) {
        /*
        page.addTargetRequests(page.getHtml().links().regex("(https://github\\.com/\\w+/\\w+)").all());
        page.putField("author", page.getUrl().regex("https://github\\.com/(\\w+)/.*").toString());
        page.putField("name", page.getHtml().xpath("//h1[@class='entry-title public']/strong/a/text()").toString());
        if (page.getResultItems().get("name")==null){
            //skip this page
            page.setSkip(true);
        }
        page.putField("readme", page.getHtml().xpath("//div[@id='readme']/tidyText()"));
    	*/
    	
    	String article=page.getHtml().xpath("/html/body/pre/text()").toString();
    	//System.out.println(article);
    	if(article!=null)
    	{
    		try {
    			fileId++;
        		out=new FileWriter("D:\\document\\web\\data\\text\\"+fileId+".txt");
        		out.write(article);
        		out.close();
    		}catch (IOException e)
    		{
    			e.printStackTrace();
    		}
    		
    	}
    	
    	//String From=page.getHtml().xpath("/html/body/table/tbody/tr/td[3]/text()").all().toString();
    	//System.out.println(From);
    	
    	/*
    	if (page.getResultItems().get("article")==null){
            page.setSkip(true);
        }
    	else {
    		page.putField("article",page.getHtml().xpath("/html/body/pre/text()"));
    	}
    	*/	
    	
    	List<String> urls = page.getHtml().xpath("/html/body/table/tbody/tr/td[4]/a/@href").all();
    	page.addTargetRequests(urls);
    }

    @Override
    public Site getSite() {
        return site;
    }

    public static void main(String[] args) {
        Spider.create(new DBWorld())
        .addUrl("https://research.cs.wisc.edu/dbworld/browse.html")
        //.addPipeline(new JsonFilePipeline("D:\\document\\web\\file"))
        .thread(5)
        .run();
    }
}