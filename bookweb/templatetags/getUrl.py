from BeautifulSoup import BeautifulSoup
from django import template

register=template.Library()

@register.filter(name='getUrl')
def getUrl(value):
    soup = BeautifulSoup(value)
    for a in soup.find_all('a', href=True):
        return(a['href'])
