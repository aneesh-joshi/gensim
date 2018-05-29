import requests
import argparse
import zipfile

"""
Utility script to download the datsets for Similarity Learning
Currently supports:
- WikiQA
- Quora Duplicate Question Pairs

Example Usage:
# To get wikiqa
$ python get_data.py --datafile wikiqa

# To get quoraqp
$ python get_data.py --datafile quoraqp
"""

# The urls and filepaths of currently supported files
wikiqa_url, wikiqa_file = "https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/", "WikiQACorpus.zip"  # noqa
quoraqp_url, quoraqp_file = "http://qim.ec.quoracdn.net/", "quora_duplicate_questions.tsv"


def download(url, file, unzip=False):
    print("Downloading %s" % file)
    req = requests.get(url + file)
    try:
        with open(file, "wb") as code:
            code.write(req.content)
            print("Download of %s complete" % file)
    except Exception as e:
        print(str(e))
    if unzip:
        print("Unzipping %s" % file)
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall()
        print("Unzip complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile',
                         default='all',
                         help='refers to the file you want to download. Options: wikiqa, quoraqp, all')
    args = parser.parse_args()

    if args.datafile == 'wikiqa':
        download(wikiqa_url, wikiqa_file, unzip=True)
    elif args.datafile == 'quoraqp':
        download(quoraqp_url, quoraqp_file)
    elif args.datafile == 'all':
        print("No arguments passed. Downloading all files.")
        download(wikiqa_url, wikiqa_file, unzip=True)
        download(quoraqp_url, quoraqp_file)
