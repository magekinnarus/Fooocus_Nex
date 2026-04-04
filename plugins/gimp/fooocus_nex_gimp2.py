#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from gimpfu import *
import urllib2
import os
import tempfile
import mimetypes
import json
import ssl

def encode_multipart_formdata(fields, files):
    boundary = '----------ThIs_Is_tHe_bouNdaRY_$'
    crlf = '\r\n'
    l = []
    for (key, value) in fields:
        l.append('--' + boundary)
        l.append('Content-Disposition: form-data; name="%s"' % key)
        l.append('')
        l.append(value)
    for (key, filename, value) in files:
        l.append('--' + boundary)
        l.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
        l.append('Content-Type: %s' % get_content_type(filename))
        l.append('')
        l.append(value)
    l.append('--' + boundary + '--')
    l.append('')
    body = crlf.join(l)
    content_type = 'multipart/form-data; boundary=%s' % boundary
    return content_type, body

def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def _urlopen(req):
    if req.get_full_url().lower().startswith('https://'):
        return urllib2.urlopen(req, context=ssl._create_unverified_context())
    return urllib2.urlopen(req)

def send_to_staging(image, layer, base_url):
    gimp.progress_init("Sending to Fooocus Staging...")
    try:
        # 1. Export active layer to temp PNG
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "gimp_to_fooocus.png")
        
        # Merge visible layers or just use active? Using active layer as requested.
        pdb.file_png_save_defaults(image, layer, temp_path, temp_path)
        gimp.progress_update(0.3)
        
        with open(temp_path, "rb") as f:
            img_data = f.read()
            
        # 2. Upload to Fooocus
        url = base_url.rstrip('/') + "/staging_api/upload"
        content_type, body = encode_multipart_formdata([], [('file', 'gimp_export.png', img_data)])
        
        req = urllib2.Request(url, data=body)
        req.add_header('Content-Type', content_type)
        req.add_header('User-Agent', 'GIMP-Fooocus-Plugin')
        
        response = _urlopen(req)
        res_data = json.load(response)
        gimp.progress_update(0.9)
        
        if res_data.get('status') == 'success':
            gimp.message("Successfully sent to Fooocus Staging: " + res_data.get('file'))
        else:
            gimp.message("Error from Fooocus: " + str(res_data.get('message', 'Unknown error')))
            
        os.remove(temp_path)
        gimp.progress_update(1.0)
        
    except Exception as e:
        gimp.message("Plugin error: " + str(e))
    finally:
        gimp.progress_update(1.0)

def receive_from_staging(image, layer, base_url):
    gimp.progress_init("Receiving from Fooocus Staging...")
    try:
        pdb.gimp_image_undo_group_start(image)
        
        # 1. Fetch targeted image from Fooocus
        url = base_url.rstrip('/') + "/staging_api/gimp_target"
        
        req = urllib2.Request(url)
        req.add_header('User-Agent', 'GIMP-Fooocus-Plugin')
        
        response = _urlopen(req)
        img_data = response.read()
        gimp.progress_update(0.5)
        
        # 2. Save to temp and load as layer
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "fooocus_to_gimp.png")
        
        with open(temp_path, "wb") as f:
            f.write(img_data)
            
        new_layer = pdb.gimp_file_load_layer(image, temp_path)
        pdb.gimp_image_insert_layer(image, new_layer, None, 0)
        
        gimp.message("Successfully received image from Fooocus Staging")
        os.remove(temp_path)
        
        pdb.gimp_image_undo_group_end(image)
        gimp.displays_flush()
        gimp.progress_update(1.0)
        
    except Exception as e:
        if image:
            pdb.gimp_image_undo_group_end(image)
        gimp.message("Plugin error: " + str(e))
    finally:
        gimp.progress_update(1.0)

register(
    "python-fu-fooocus-nex-send",
    "Send active layer to Fooocus Staging",
    "Sends the current active layer as a PNG to the Fooocus_Nex staging area.",
    "Antigravity", "Antigravity", "2026",
    "Send to Staging...",
    "RGB*, GRAY*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_STRING, "base_url", "Fooocus API URL", "http://localhost:7865")
    ],
    [],
    send_to_staging,
    menu="<Image>/Filters/Fooocus_Nex"
)

register(
    "python-fu-fooocus-nex-receive",
    "Receive image from Fooocus Staging",
    "Receives the currently targeted image from Fooocus_Nex staging area as a new layer.",
    "Antigravity", "Antigravity", "2026",
    "Receive from Staging...",
    "RGB*, GRAY*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_STRING, "base_url", "Fooocus API URL", "http://localhost:7865")
    ],
    [],
    receive_from_staging,
    menu="<Image>/Filters/Fooocus_Nex"
)

main()
