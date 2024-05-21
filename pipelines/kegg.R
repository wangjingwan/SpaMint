# .libPaths('D:/0.work/rlib/')
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")

# BiocManager::install("org.Hs.eg.db")
# BiocManager::install("clusterProfiler")
# TODO
##### TODO this script was performed on 620 cluster #####
args <- commandArgs(trailingOnly = TRUE)
.libPaths('/home/wangjingwan/R/x86_64-pc-linux-gnu-library/4.2')
suppressPackageStartupMessages(library(clusterProfiler))

# setwd('/data6/wangjingwan/5.Simpute/4.datasets/1.mela/spex/spa')
# setwd('/data6/wangjingwan/5.Simpute/4.datasets/1.mela/spa/sp_kegg/')
# setwd('/data6/wangjingwan/5.Simpute/4.datasets/BR_c2l_S1_rep50/spade/')
setwd(args[1])
if (args[2] == 'human') {
  suppressPackageStartupMessages(library(org.Hs.eg.db))
  ref_db <- org.Hs.eg.db
  org = 'hsa'
} else if (args[2] == 'mouse') {
  suppressPackageStartupMessages(library(org.Mm.eg.db))
  ref_db <- org.Mm.eg.db
  org = 'mmu'
}


if (length(args) > 2) {
  fn_lst = c(args[3])
}else{
  fn_lst = list.files('./', pattern = "kegg.tsv")
}


for (fn in fn_lst) {
  print(fn)
  tryCatch({
    top <- read.table(file = fn, sep = '\t', header = TRUE, row.names = 1)
    new_file_name <- paste0(sub("\\.tsv$", "", fn), "_enrichment", ".tsv")
    gene <- bitr(top$gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = ref_db)
    gene$logFC <- top$lr_co_exp_num[match(gene$SYMBOL, top$gene)]
    write.table(gene, file = paste0(sub("\\.tsv$", "", fn), "_geneID", ".tsv"), sep = '\t', quote = FALSE)
    geneList <- gene$logFC
    names(geneList) <- gene$ENTREZID
    geneList <- sort(geneList, decreasing = TRUE)
    head(geneList)
    gene <- names(geneList)[abs(geneList) > 0]
    
    kk <- enrichKEGG(gene = gene, organism = org, pvalueCutoff = 0.05)
    
    write.table(kk@result, file = new_file_name, sep = '\t', quote = FALSE)
    pdf(file = paste0(new_file_name, ".pdf"), width = 8, height = 10)
    dotplot(kk, showCategory = 30)
    dev.off()
  }, error = function(e) {
    # Handle the error here or print an error message
    print(paste("Error occurred for file:", fn))
  })
}