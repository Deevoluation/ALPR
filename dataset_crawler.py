# make sure to install all the dependencies prior running the script. happy hacking xD

'''
  author @DravitLochan
'''
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "http://www.numberplates.com/pr-number-plate-gallery.asp?page="
urlSite = "www.numberplates.com/"


def main():
    f = open("data_set_urls.txt", "w")
    # urls = []
    for page_num in range(510, 0, -1):
        # url = url + str(page_num)
        page = urlopen(url + str(page_num))
        soup = BeautifulSoup(page)
        all_anchors = soup.find_all('a')
        for anchor in all_anchors:
            if anchor.has_attr('name'):
                img = anchor.find_all('img')
                if img[0].has_attr('src'):
                    print(urlSite + str(img[0]['src']))
                    # urls.append(urlSite + str(img[0]['src']))
                    f.write(urlSite + str(img[0]['src']) + "\n")

    # f.write(urls)
    f.close()


if __name__ == "__main__":
    main()
