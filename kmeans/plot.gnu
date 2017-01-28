GNUTERM = "x11"
set term pdfcairo
set output "kmeans_plot.pdf"

set title "k means"
set palette model RGB defined (0 "red",1 "blue",2 "green")
plot 'output.txt' using 1:2:3 with points pt 3 palette

set output "kmeans_distortion.pdf"
set title 'k means distortion'
set style data lines
plot 'kmeans_distortion.txt' using 1:2

set output "kmeans++_distortion.pdf"
set title 'k means++ distortion'
plot 'kmeans++_distortion.txt' using 1:2
