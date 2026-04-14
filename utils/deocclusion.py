'''
release version
'''
import base64
import json
import os
import requests

headers = {'Content-Type': 'application/json'}

'''
# fastapi document:
url_1 = "http://lightions.ai:44731/docs"
url_2 = "http://lightions.ai:44318/docs"
url_3 = "http://lightions.ai:44424/docs"
url_4 = "http://lightions.ai:44075/docs"
# check health:
url_1 = "http://lightions.ai:44731/health"
url_2 = "http://lightions.ai:44318/health"
url_3 = "http://lightions.ai:44424/health"
url_4 = "http://lightions.ai:44075/health"
# call deocclution local(local/nas/jfs):
url_1 = "http://lightions.ai:44731/deocclusion_local"  # render-25
url_2 = "http://lightions.ai:44318/deocclusion_local"  # render-27
url_3 = "http://lightions.ai:44424/deocclusion_local"  # render-29
url_4 = "http://lightions.ai:44075/deocclusion_local"  # render-13
# call deocclution network(base64):
url_1 = "http://lightions.ai:44731/deocclusion_network"
url_2 = "http://lightions.ai:44318/deocclusion_network"
url_3 = "http://lightions.ai:44424/deocclusion_network"
url_4 = "http://lightions.ai:44075/deocclusion_network"
'''
def call_deocclution_network(url, dual_image_path, prompt, output_image_path, seed=None):
    # with open(prompt_path, 'r') as f:
    #     prompt = f.read()
    # prompt = "bear"
    with open(dual_image_path, 'rb') as f:
        dual_image_bytes = base64.b64encode(f.read()).decode('utf-8')
    request = {
        "dual_image_bytes": dual_image_bytes,
        "dual_image_ext": os.path.splitext(dual_image_path)[1],
        "prompt": prompt,
        "output_image_ext": os.path.splitext(output_image_path)[1],
        "seed": seed,
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(request)).json()
    output_image_bytes = response['output_image_bytes']
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    with open(output_image_path, 'wb') as f:
        f.write(base64.b64decode(output_image_bytes))
        

def deocclusion(dual_image_path, prompt, output_image_path, seed=None, \
                url="http://lightions.ai:44731/deocclusion_network"):
    with open(dual_image_path, 'rb') as f:
        dual_image_bytes = base64.b64encode(f.read()).decode('utf-8')
    request = {
        "dual_image_bytes": dual_image_bytes,
        "dual_image_ext": os.path.splitext(dual_image_path)[1],
        "prompt": prompt,
        "output_image_ext": os.path.splitext(output_image_path)[1],
        "seed": seed,
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(request)).json()
    output_image_bytes = response['output_image_bytes']
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    with open(output_image_path, 'wb') as f:
        f.write(base64.b64decode(output_image_bytes))


if __name__ == '__main__':
    url = "http://lightions.ai:44731/deocclusion_network"
    fname="sofa_8930"
    dual_image_path = f"data/occlusion/{fname}.png"
    prompt = f"sofa"
    output_image_path = f"outputs/viz/occlusion/{fname}.png"
    call_deocclution_network(url, dual_image_path, prompt, output_image_path, seed=666)
    
'''
python utils/deocclusion.py
'''

