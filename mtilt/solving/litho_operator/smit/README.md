## How to test GWX smit simulator

1. generate .src source file:
```shell
cd ./gensource
python gensource.py ${SOURCEFILE.src}
```

2. generate .tcc files:
```shell
cd ./gentcc
python gentcc.py ${TCCSAVEPATH} ${SOURCEFILE.src}
```

3. generate image/mask:
```shell
cd ./genimage
python genimage.py ${TCCSAVEPATH}, ${LAYOUT.gds}
```