import scrapy

class FirstSpider(scrapy.Spider):

    name = "FirstSpider"
        
    def start_request(self):
        urls = ['http://quotes.toscrape.com/']
        for url in urls:
            yield scrapy.Request(url= url, callback= self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

