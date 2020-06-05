import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('bib')
parser.add_argument('aux')
args = parser.parse_args()

with open(args.bib) as fp:
    text = fp.read()
    available_keys = set(key.strip() for key in re.findall(r'@\w+{([^-].*?),', text))

with open(args.aux) as fp:
    text = fp.read()
    used_keys = set(key.strip() for key in re.findall(r'\\abx@aux@cite\{(.*?)\}', text))

print(f'found {len(available_keys)} available keys')
print(f'found {len(used_keys)} used keys')
print('unused keys:')
print('\n'.join(sorted(available_keys - used_keys)))
