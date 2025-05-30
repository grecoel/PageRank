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
    total_pages = len(corpus)
    probabilities = dict()

    if corpus[page]:
        linked_pages = corpus[page]
        num_links = len(linked_pages)
        for p in corpus:
            probabilities[p] = (1 - damping_factor) / total_pages
            if p in linked_pages:
                probabilities[p] += damping_factor / num_links
    else:
        # If no links, treat as linking to all pages
        for p in corpus:
            probabilities[p] = 1 / total_pages

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    """
    page_counts = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_counts[page] += 1
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(
            population=list(model.keys()),
            weights=list(model.values()),
            k=1
        )[0]

    # Normalize counts to sum to 1
    pageranks = {page: count / n for page, count in page_counts.items()}
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    convergence_threshold = 0.001

    # Handle pages with no links: treat them as linking to all pages
    no_link_pages = {page for page, links in corpus.items() if not links}

    while True:
        new_rank = {}
        for page in corpus:
            total = 0
            for potential_linker in corpus:
                links = corpus[potential_linker]
                if not links:
                    total += pagerank[potential_linker] / N
                elif page in links:
                    total += pagerank[potential_linker] / len(links)
            new_rank[page] = (1 - damping_factor) / N + damping_factor * total

        # Check convergence
        if all(abs(new_rank[page] - pagerank[page]) < convergence_threshold for page in pagerank):
            break

        pagerank = new_rank.copy()

    # Normalize to sum to 1
    total_sum = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_sum

    return pagerank



if __name__ == "__main__":
    main()
