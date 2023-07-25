import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    linked_pages = corpus[page]
    num_pages = len(corpus)
    probability_distribution = {}
    if linked_pages:
        probability_linked = damping_factor / len(linked_pages)
        for p in corpus:
            probability_distribution[p] = probability_linked if p in linked_pages else 0
    else:
        probability_linked = damping_factor / num_pages
        for p in corpus:
            probability_distribution[p] = probability_linked
    probability_random = (1 - damping_factor) / num_pages
    for p in corpus:
        probability_distribution[p] += probability_random
    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))
    pagerank[current_page] += 1

    for _ in range(n - 1):
        transition_probs = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(list(transition_probs.keys()), list(transition_probs.values()))[0]
        pagerank[next_page] += 1
        current_page = next_page
    total_samples = n
    
    for page in pagerank:
        pagerank[page] /= total_samples
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    pagerank = {page: 1 / num_pages for page in corpus}
    damping_component = (1 - damping_factor) / num_pages

    while True:
        new_pagerank = {}
        for page in corpus:
            pr_sum = sum(pagerank[link] / len(corpus[link]) for link in corpus if page in corpus[link])
            new_pagerank[page] = damping_component + damping_factor * pr_sum

        if all(abs(new_pagerank[page] - pagerank[page]) < 0.001 for page in pagerank):
            break
        pagerank = new_pagerank
    return pagerank


if __name__ == "__main__":
    main()
