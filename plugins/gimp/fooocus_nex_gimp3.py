#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('GObject', '2.0')
from gi.repository import GObject
gi.require_version('Gio', '2.0')
from gi.repository import Gio
gi.require_version('GLib', '2.0')
from gi.repository import GLib

import urllib.request
import urllib.parse
import os
import tempfile
import json
import ssl

def _urlopen(req):
    if req.full_url.lower().startswith('https://'):
        return urllib.request.urlopen(req, context=ssl._create_unverified_context())
    return urllib.request.urlopen(req)

class FooocusNexStaging(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "base_url": (str, "Base URL", "Fooocus API URL", "http://localhost:7865", GObject.ParamFlags.READWRITE),
    }

    def do_query_procedures(self):
        return ["fooocus-nex-send", "fooocus-nex-receive"]

    def do_create_procedure(self, name):
        procedure = Gimp.Procedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
        
        if name == "fooocus-nex-send":
            procedure.set_documentation("Send to Staging", "Sends the current active layer as a PNG to the Fooocus_Nex staging area.", name)
            procedure.set_menu_label("Send to Staging...")
        else:
            procedure.set_documentation("Receive from Staging", "Receives the currently targeted image from Fooocus_Nex staging area as a new layer.", name)
            procedure.set_menu_label("Receive from Staging...")
            
        procedure.add_menu_path("<Image>/Filters/Fooocus_Nex")
        procedure.set_image_types("RGB*, GRAY*")
        
        procedure.add_argument(Gimp.ExportProcedure.create_image_argument(procedure, "image", True))
        procedure.add_argument(Gimp.ExportProcedure.create_drawable_argument(procedure, "drawable", True))
        procedure.add_argument(GObject.Property.new_str("base_url", "Base URL", "Fooocus API URL", "http://localhost:7865", GObject.ParamFlags.READWRITE))
        
        return procedure

    def run(self, procedure, args, data):
        name = procedure.get_name()
        img = args.index(0)
        drawable = args.index(1)
        base_url = args.index(2)

        if name == "fooocus-nex-send":
            return self.send_to_staging(img, drawable, base_url)
        else:
            return self.receive_from_staging(img, drawable, base_url)

    def encode_multipart(self, filename, file_content):
        boundary = '----------ThIs_Is_tHe_bouNdaRY_$'
        crlf = b'\r\n'
        body = []
        body.append(b'--' + boundary.encode())
        body.append(b'Content-Disposition: form-data; name="file"; filename="' + filename.encode() + b'"')
        body.append(b'Content-Type: image/png')
        body.append(b'')
        body.append(file_content)
        body.append(b'--' + boundary.encode() + b'--')
        body.append(b'')
        return 'multipart/form-data; boundary=%s' % boundary, b'\r\n'.join(body)

    def send_to_staging(self, image, drawable, base_url):
        Gimp.Progress.init("Sending to Fooocus Staging...")
        try:
            # 1. Export to temp
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "gimp3_to_fooocus.png")
            
            # GIMP 3.0 API for saving
            Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, drawable, Gio.File.new_for_path(temp_path))
            Gimp.Progress.update(0.3)
            
            with open(temp_path, "rb") as f:
                img_data = f.read()
                
            # 2. Upload
            url = base_url.rstrip('/') + "/staging_api/upload"
            content_type, body = self.encode_multipart("gimp3_export.png", img_data)
            
            req = urllib.request.Request(url, data=body)
            req.add_header('Content-Type', content_type)
            req.add_header('User-Agent', 'GIMP3-Fooocus-Plugin')
            
            with _urlopen(req) as response:
                res_data = json.loads(response.read().decode())
            Gimp.Progress.update(0.9)
            
            if res_data.get('status') == 'success':
                print("Successfully sent to Fooocus Staging")
            
            os.remove(temp_path)
            Gimp.Progress.update(1.0)
            return self.procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error.new_literal(Gimp.PlugIn.getErrorQuark(), 0, ""))
        except Exception as e:
            print("Plugin error:", str(e))
            Gimp.Progress.update(1.0)
            return self.procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, None)

    def receive_from_staging(self, image, drawable, base_url):
        Gimp.Progress.init("Receiving from Fooocus Staging...")
        try:
            image.undo_group_start()
            
            # 1. Fetch
            url = base_url.rstrip('/') + "/staging_api/gimp_target"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'GIMP3-Fooocus-Plugin')
            
            with _urlopen(req) as response:
                img_data = response.read()
            Gimp.Progress.update(0.5)
                
            # 2. Save and load
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "fooocus_to_gimp3.png")
            with open(temp_path, "wb") as f:
                f.write(img_data)
                
            new_layer = Gimp.file_load_layer(Gimp.RunMode.NONINTERACTIVE, image, Gio.File.new_for_path(temp_path))
            image.insert_layer(new_layer, None, 0)
            
            os.remove(temp_path)
            image.undo_group_end()
            Gimp.displays_flush()
            Gimp.Progress.update(1.0)
            
            return self.procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)
        except Exception as e:
            if image:
                image.undo_group_end()
            print("Plugin error:", str(e))
            Gimp.Progress.update(1.0)
            return self.procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, None)

Gimp.main(FooocusNexStaging.__gtype__, [])
