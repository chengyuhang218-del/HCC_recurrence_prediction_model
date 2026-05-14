#下载github数据后，记得改为自己的路径

library(GEOquery)

gset <- getGEO(
  filename = "/Users/chengyuhang/Desktop/GSE76427_series_matrix.txt",
  AnnotGPL = FALSE,
  getGPL = FALSE)

exprSet <- exprs(gset)   # 表达矩阵：probe × sample
pdata <- pData(gset)     # 临床信息

colnames(pdata)
dim(exprSet)

unique(pdata$`tissue:ch1`)
unique(pdata$`event_rfs:ch1`)

rfs_label <- data.frame(
  sample = rownames(pdata),
  tissue = pdata$`tissue:ch1`,
  recurrence = pdata$`event_rfs:ch1`)

rfs_label <- rfs_label[
  rfs_label$tissue == "primary hepatocellular carcinoma tumor",]

rfs_label$tissue <- NULL

rfs_label <- rfs_label[!is.na(rfs_label$recurrence), ]
rfs_label <- rfs_label[rfs_label$recurrence != "NA", ]
rfs_label$recurrence <- as.numeric(rfs_label$recurrence)

table(rfs_label$recurrence)
dim(rfs_label)

tumor_samples <- rfs_label$sample
expr_tumor <- exprSet[, tumor_samples]

dim(expr_tumor)
dim(rfs_label)
all(colnames(expr_tumor) == rfs_label$sample)


anno <- read.delim(
  "/Users/chengyuhang/Desktop/GPL10558_HumanHT-12_V4_0_R1_15002873_B.txt",
  skip = 8,
  header = TRUE,
  sep = "\t",
  quote = "",
  fill = TRUE,
  check.names = FALSE)

head(anno[, c("Probe_Id", "Symbol")])


probe2gene <- anno[, c("Probe_Id", "Symbol")]
probe2gene <- probe2gene[match(rownames(expr_tumor), probe2gene$Probe_Id),]
rownames(expr_tumor) <- probe2gene$Symbol
head(rownames(expr_tumor))

expr_tumor <- expr_tumor[rownames(expr_tumor) != "" & !is.na(rownames(expr_tumor)),]


probe_var <- apply(expr_tumor, 1, var)
expr_tumor_ordered <- expr_tumor[order(probe_var, decreasing = TRUE), ]
expr_final <- expr_tumor_ordered[!duplicated(rownames(expr_tumor_ordered)), ]

sum(duplicated(rownames(expr_final)))
dim(expr_final)

expr_final <- as.matrix(expr_final)
mode(expr_final) <- "numeric"

cat("NA count before clean:", sum(is.na(expr_final)), "\n")
cat("Inf count before clean:", sum(is.infinite(expr_final)), "\n")

expr_final[is.infinite(expr_final)] <- NA
expr_final <- expr_final[complete.cases(expr_final), ]

cat("NA count after clean:", sum(is.na(expr_final)), "\n")
cat("Inf count after clean:", sum(is.infinite(expr_final)), "\n")

dim(expr_final)

# 这是处理好的用于训练最终数据，也在github有存
write.csv(expr_final,"/Users/chengyuhang/Desktop/GSE76427_expr_gene_RFS_108_clean.csv")
# 这是标签
write.csv(rfs_label,"/Users/chengyuhang/Desktop/GSE76427_rfs_label_108.csv",row.names = FALSE)