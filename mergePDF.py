import PyPDF2

def merge_pdfs(paths, output):
    pdf_writer = PyPDF2.PdfWriter()

    for path in paths:
        pdf_reader = PyPDF2.PdfReader(path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(output, 'wb') as out:
        pdf_writer.write(out)

# Przykładowe użycie
paths = ['Zadanie5.pdf', 'Zadanie6.pdf', 'Zadanie7.pdf', 'Zadanie8.pdf', 'Zadanie9.pdf', 'Zadanie10.pdf', 'Zadanie11.pdf', 'Zadanie12.pdf']
merge_pdfs(paths, 'combined.pdf')
