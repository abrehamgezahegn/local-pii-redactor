"""Opt-in real-Ollama regression suite. Run as a script, not in normal pytest."""
import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pii_redactor.core import Box, Entity, entity_regions, load_image, ocr, png_bytes, redact
from pii_redactor.pipeline import analyze

FIELDS = [("Name", "Maya Tesfaye"), ("Email", "maya.tesfaye@example.com"), ("Phone", "+1 (202) 555-0147"), ("Address", "184 Cedar Lane, Portland, OR 97205"), ("Date of birth", "1992-04-18"), ("Passport", "AB1234567")]


def fixture(font_size=28, fg="black", bg="white"):
    image=Image.new("RGB",(1400,650),bg)
    font=ImageFont.load_default(size=font_size)
    draw=ImageDraw.Draw(image)
    masks=[]
    for i,(label,value) in enumerate(FIELDS):
        y=40+i*90
        draw.text((35,y),label+":",font=font,fill=fg)
        draw.text((250,y),value,font=font,fill=fg)
        mask=Image.new("L",image.size,0)
        ImageDraw.Draw(mask).text((250,y),value,font=font,fill=255)
        masks.append(mask)
    return image,masks


def variants():
    base,masks=fixture()
    yield "standard",base,masks
    for name,size,fg,bg in [("small12",12,"black","white"),("tiny9",9,"black","white"),("dark",28,"white","black"),("low_contrast",28,"#c5c5c5","#dddddd")]:
        image,mm=fixture(size,fg,bg);yield name,image,mm
    for angle in (90,180,270,4,-4):
        yield f"rotation_{angle}",base.rotate(angle,expand=True,fillcolor="white"),[m.rotate(angle,expand=True,fillcolor=0) for m in masks]
    yield "blur",base.filter(ImageFilter.GaussianBlur(.7)),masks
    for fmt,options in [("JPEG",{"quality":30}),("WEBP",{"quality":70}),("HEIF",{"quality":70}),("AVIF",{"quality":60}),("GIF",{}),("BMP",{})]:
        data=io.BytesIO();base.save(data,format=fmt,**options)
        yield fmt.lower(),load_image(data.getvalue()),masks
    yield "grayscale",base.convert("L").convert("RGB"),masks
    high=base.convert("F").point(lambda p:p*100+30000)
    data=io.BytesIO();high.save(data,format="TIFF")
    yield "high_bit_tiff",load_image(data.getvalue()),masks
    alpha=base.convert("RGBA")
    alpha.putalpha(base.convert("L").point(lambda p:255-p))
    data=io.BytesIO();alpha.save(data,format="PNG")
    yield "transparent",load_image(data.getvalue()),masks
    for name,size,position in [("tall",(1400,5000),(0,4200)),("wide",(6000,650),(4600,0))]:
        image=Image.new("RGB",size,"white");image.paste(base,(0,0));image.paste(base,position)
        mm=[]
        for mask in masks:
            enlarged=Image.new("L",size,0);enlarged.paste(mask,(0,0));enlarged.paste(mask,position);mm.append(enlarged)
        yield name,image,mm


def coverage_checks(image,masks,regions):
    boxes=[Box(**b) for r in regions for b in r["boxes"]]
    coverage=redact(Image.new("RGB",image.size,"white"),boxes,5).convert("L") if boxes else Image.new("L",image.size,255)
    cover=coverage.tobytes()
    fields=[]
    for (label,_),mask in zip(FIELDS,masks):
        targets=mask.tobytes()
        ink=sum(p>48 for p in targets)
        missed=sum(p>48 and c!=0 for p,c in zip(targets,cover))
        fields.append({"field":label,"coverage":round(1-missed/max(1,ink),4),"uncovered_ink_pixels":missed})
    return fields,boxes


def run(args):
    results=[]
    for name,image,masks in variants():
        if args.cases and name not in args.cases.split(","):
            continue
        started=time.monotonic()
        result=analyze(image,"llama3.2:latest")
        fields,boxes=coverage_checks(image,masks,result["regions"])
        record={"case":name,"complete":result["complete"],"regions":len(result["regions"]),"fields":fields,"pass":result["complete"] and all(f["uncovered_ink_pixels"]==0 for f in fields),"seconds":round(time.monotonic()-started,1),"warnings":result["warnings"]}
        results.append(record); print(json.dumps(record),flush=True)
        Path(args.report).write_text(json.dumps(results,indent=2))
    if not args.cases:
        for name,image in [("blank",Image.new("RGB",(600,400),"white")),("non_pii",Image.new("RGB",(1000,300),"white"))]:
            if name=="non_pii":
                draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=28)
                for i,text in enumerate(["Monthly report", "Total items: 42", "Meeting date: 2026-11-18"]):draw.text((30,30+i*70),text,font=font,fill="black")
            result=analyze(image,"llama3.2:latest")
            record={"case":name,"pass":not result["regions"],"complete":result["complete"],"regions":len(result["regions"])}
            results.append(record);print(json.dumps(record),flush=True)
    if args.screenshot:
        image=load_image(Path(args.screenshot).read_bytes())
        result=analyze(image,"llama3.2:latest")
        words=ocr(image)
        expected=["Andrej Karpathy","Fei-Fei Li","Daphne Koller","Andrew Ng","Sebastian Thrun","Vladlen Koltun","Koray Kavukcuoglu","Vlad Mnih","Michiel van de Panne","Geoff Hinton"]
        boxes=[Box(**b) for r in result["regions"] for b in r["boxes"]]
        checked=[]
        for name in expected:
            regions,unmatched=entity_regions(words,[Entity(text=name,category="name")])
            covered=bool(regions) and all(any(b.x<=target.x and b.y<=target.y and b.x+b.width>=target.x+target.width and b.y+b.height>=target.y+target.height for b in boxes) for region in regions for target in region.boxes)
            checked.append({"name":name,"covered":covered})
        record={"case":"user_screenshot","pass":result["complete"] and all(c["covered"] for c in checked),"complete":result["complete"],"regions":len(result["regions"]),"checks":checked,"warnings":result["warnings"]}
        results.append(record);print(json.dumps(record),flush=True)
        if boxes:
            Path(args.report).with_name("screenshot-redacted.png").write_bytes(png_bytes(redact(image,boxes,5)))
    Path(args.report).write_text(json.dumps(results,indent=2))
    failed=[r["case"] for r in results if not r["pass"]]
    print(json.dumps({"passed":len(results)-len(failed),"total":len(results),"failed":failed}),flush=True)
    return bool(failed)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",required=True)
    parser.add_argument("--screenshot")
    parser.add_argument("--cases")
    raise SystemExit(run(parser.parse_args()))
