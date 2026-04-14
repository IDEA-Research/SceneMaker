`watertight_and_sampling.py` is a data preprocessing module designed for shape VAE and diffusion models. 
It implements advanced watertight mesh processing techniques utilizing both depth testing and winding number 
calculations for robust watertight conversion used in [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D) 
and sharp edge samling strategy proposed in [Dora](https://github.com/Seed3D/Dora).

```
python watertight_and_sampling.py --input_mesh input.obj --skip_watertight
```
