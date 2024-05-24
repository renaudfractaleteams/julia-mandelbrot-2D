#pandoc -t markdown_strict --extract-media='./finality'  -s README_MAIN.md 01_Create_Datas/README.md -o README.md
pandoc -t markdown_strict  ./00_source_doc/main_temp_en.docx -o READM_temp.md
pandoc --dpi=600 README.md  -o ./00_source_doc/main_temp.docx