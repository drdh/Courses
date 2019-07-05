import scrapy
import tomd
import os

class ArchiveSpider(scrapy.Spider):
    name="archive"

    def start_requests(self):
        urls=[
            "http://www.ruanyifeng.com/blog/archives.html",
        ]

        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse)


    def parse(self,response):
        urls=response.xpath('//div[@id="beta"]//li/a/@href').extract()
        titles=response.xpath('//div[@id="beta"]//li/a/@title').extract()

        for href in urls:
            yield response.follow(href,self.parse_cate)
        

    def parse_cate(self,response):
        urls=response.xpath('//div[@id="alpha"]//li/a/@href').extract()
        
        for url in urls:
            yield response.follow(url,self.parse_blog)

    def parse_blog(self,response):
        article=response.xpath('//article/div[@class="asset-content entry-content"]').extract()
        mdTxt=tomd.Tomd(article[0]).markdown
        name=response.xpath("//article/h1/text()").extract()[0]
        cate=response.xpath('//div[@class="entry-categories"]//li/a/text()').extract()[0]

        path=os.path.expandvars('$HOME')+"/Downloads/ruanyifeng/"+cate+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path+name+".md","w+") as f:
            f.write(mdTxt)
            f.close()
        


