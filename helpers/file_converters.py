__author__ = 'suraj'


from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import codecs
import docx
import os, subprocess
from config import Config


__all__ = ['pdf2txt', 'doc2txt', 'txt2txt']



def pdf2txt(fname, skip=None):
    print fname
    env = dict(os.environ)
    env['LC_ALL'] = 'en_US.UTF-8'
    out_fname= os.path.splitext(os.path.basename(fname))[0]
    cmd = ["pdftotext",fname,os.path.join(Config.PDF_PARSE,out_fname+'.txt') ]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    out,err = p.communicate()
    return txt2txt(os.path.join(Config.PDF_PARSE,out_fname+'.txt'))






#def pdf2txt(fname, skip=None):
#    output = StringIO()
#    manager = PDFResourceManager()
#    codec = 'utf-8'
#    converter = TextConverter(manager, output, codec=codec, laparams=LAParams())
#    interpreter = PDFPageInterpreter(manager, converter)

#    infile = file(fname, 'rb')
#
#    for page_id, page in enumerate(PDFPage.get_pages(infile)):
#        if skip and page_id < skip:
#            continue
#        interpreter.process_page(page)
#    infile.close()
#    converter.close()
#    text = output.getvalue()
#    output.close
#    return text.replace('\r', '').replace('\x0C', '').decode('utf-8')


def doc2txt(fname, skip=None):
    doc = docx.Document(fname)

    fullText = []
    for para_id, para in enumerate(doc.paragraphs):
        if skip and  para_id<skip:
            continue
        fullText.append(para.text.encode('utf-8'))
    return '\n'.join(fullText)


# TODO: need to handle skip
def txt2txt(fname, skip=None):
    """Read the text file

    """
    with codecs.open(fname, 'r', encoding='utf-8') as f_in:
        content = f_in.read()

	content.replace('\r', '').replace('\x0C', '')

    return content


if __name__ == '__main__':
    preprocess = lambda x: x.replace('\n', ' ').replace('\r', '').replace('\x0C', '')
    doc_file = '/home/suraj/test_booxby/BPA.docx'

    pdf_file = '/home/suraj/test_booxby/Sugerland.pdf'
    # print preprocess(pdf2txt(pdf_file,skip=6))
    print doc2txt(doc_file)
