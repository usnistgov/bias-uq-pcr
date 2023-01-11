file := bias-uq-pcr
out_dir := out


run:
	python3 src/plot_fig2.py
	python3 src/plot_fig4_write_f_cw.py
	python3 src/plot_fig5.py

copy:
	cp -a out/Fig*.png ../paper/fig/
	cp -a out/*.npy ../paper/out/