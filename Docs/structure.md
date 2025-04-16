## File description


```
.      
├─benchmark
│  └─ICCAD2013 
│          *.glp
├─configs (config files)
│  └─ICCAD2013
│          *.yaml       
├─Docs
│      API_Documentation.md
│      introductions.md
│      
├─imgs
├─mtilt: main package)
│  │  
│  ├─config: build config args
│  ├─data: build masks dataloader for optimization      
│  ├─engine: basic optimization engine
│  ├─evaluation: evaluate results **NOTE**: but still some bugs
│  ├─optimizer
│  │  ├─base_optimizer
│  │  └─weighting: mutil objective optimization methods
│  │          
│  ├─solving
│  │  ├─algorithms: optimization
│  │  ├─intializer: levelset & pixel-style mask intialization
│  │  ├─litho_operator: lithography simulators          
│  │  ├─objectives    
│  │  └─solver: basic solver        
│  └─utils: some useful tools
|  
├─thirdparty
│  └─adaptive-boxes: shot counting tool
|
└─tools
        *.py
        solve.py: main
```