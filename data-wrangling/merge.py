import nbformat
from nbconvert import MarkdownExporter

# List of your notebooks in the order you want them merged
notebooks = ['Ex1.ipynb', 'Ex2.ipynb', 'Ex3.ipynb']
merged_nb = nbformat.v4.new_notebook()

# Merge cells from all notebooks
for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        merged_nb.cells.extend(nb.cells)

# Export to Markdown
md_exporter = MarkdownExporter()
(body, resources) = md_exporter.from_notebook_node(merged_nb)

with open('final_document.md', 'w', encoding='utf-8') as f:
    f.write(body)
