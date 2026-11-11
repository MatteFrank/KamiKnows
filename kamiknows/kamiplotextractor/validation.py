"""Source overlays and explicit, independent-reference numerical comparisons."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from statistics import median

from .digitization import Calibration, finite


def structural_check(series: dict) -> None:
    x,y=series.get("x",[]),series.get("y",[])
    if not x or len(x)!=len(y) or not series.get("series_id"):
        raise ValueError("empty/misaligned numerical series")
    for value in [*x,*y]:
        if value is not None:
            finite(value)
    if any(v is None for v in x):
        raise ValueError("x coordinates must be explicit")
    if any(v is None for v in y) and series.get("geometry")!="band_envelopes":
        raise ValueError("missing central values require explicit band geometry")
    for field in ("bins","displayed_uncertainties","value_states"):
        if series.get(field) and len(series[field])!=len(x):
            raise ValueError(f"misaligned {field}")
    for bounds in series.get("bins") or []:
        if len(bounds)!=2 or finite(bounds[0])>=finite(bounds[1]):
            raise ValueError("invalid bin boundaries")
    for component in series.get("displayed_uncertainties",[]):
        for key in ("lower","upper","magnitude"):
            if component.get(key) is not None:
                finite(component[key])


def overlay(record: dict, work_dir: Path) -> dict:
    """SVG embeds a local source raster; marks are in original source coordinates."""
    from PIL import Image

    source=work_dir/"source.png"
    if not source.is_file():
        return {"status":"unavailable","reason":"source rendering unavailable; no overlay claimed"}
    with Image.open(source) as image:
        width,height=image.size
    scale=record["raw_source"].get("render_scale",1)
    data=base64.b64encode(source.read_bytes()).decode("ascii")
    elements=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              f'<title>{html.escape(record["data_id"])} — manual review pending</title>',
              f'<image width="{width}" height="{height}" href="data:image/png;base64,{data}"/>']
    def trace(points: list, color: str) -> None:
        encoded=" ".join(f"{finite(x)*scale},{finite(y)*scale}" for x,y in points)
        elements.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    for i,series in enumerate(record["series"]):
        points=series.get("source_coordinates",[])
        color=("#ff00ff","#00aacc","#dd5500")[i%3]
        if series["geometry"]=="rendered_curve_samples":trace(points,color)
        elif series["geometry"]=="band_envelopes":
            for envelope in series["source_envelopes"]:trace(envelope,color)
        else:
            for x,y in points:
                elements.append(f'<circle cx="{finite(x)*scale}" cy="{finite(y)*scale}" r="3" fill="none" stroke="{color}"/>')
        cal=Calibration(record["calibration"])
        for j,component in enumerate(series.get("displayed_uncertainties",[])):
            if component.get("lower") is not None and component.get("upper") is not None:
                x=cal.x.position(series["x"][j])
                trace([[x,cal.y.position(component["lower"])],[x,cal.y.position(component["upper"])]],color)
        for j,bounds in enumerate(series.get("bins",[])):
            left,right=map(cal.x.position,bounds)
            top,bottom=cal.y.position(series["y"][j]),cal.y.position(series["baseline"])
            trace([[left,bottom],[left,top],[right,top],[right,bottom]],color)
    elements.append('</svg>')
    target=work_dir/"overlay.svg"
    target.write_text("\n".join(elements),encoding="utf-8")
    return {"status":"generated_manual_review_pending","path":str(target.name),"kind":"source_geometry_overlay",
            "scientific_validity":"not_established","data_id":record["data_id"]}


def _summary(values: list[float]) -> dict:
    ordered=sorted(values)
    return {"count":len(ordered),"median":median(ordered) if ordered else None,
            "p95":ordered[max(0,math.ceil(.95*len(ordered))-1)] if ordered else None,
            "max":max(ordered) if ordered else None}


def compatible(result: dict, reference: dict) -> bool:
    return all(result.get(key) is not None and result.get(key)==reference.get(key)
               for key in ("series_id","observable","coordinate_definition","units"))


def compare_series(result: dict, reference: dict, comparison: dict) -> dict:
    """No row alignment: explicit point identity, unique nearest points or stated interpolation."""
    structural_check(result);structural_check(reference)
    if not compatible(result,reference):
        return {"status":"not_comparable","reason":"observable, units, series or coordinate definition differs/missing"}
    if bool(result.get("bins")) != bool(reference.get("bins")):
        return {"status":"not_comparable","reason":"binned and unbinned observables cannot be paired"}
    tolerance=comparison.get("tolerance",{})
    tx,ty=finite(tolerance.get("x",0)),finite(tolerance.get("y",0))
    if min(tx,ty)<=0:
        raise ValueError("comparison requires positive physical x/y tolerances")
    method=comparison.get("method","unique_nearest")
    pairs=[];ambiguous=0
    rx,ry=result["x"],result["y"]
    gx,gy=reference["x"],reference["y"]
    interpolated=None
    if method=="point_ids":
        a,b=result.get("point_ids",[]),reference.get("point_ids",[])
        if len(a)!=len(rx) or len(b)!=len(gx) or len(set(a))!=len(a) or len(set(b))!=len(b):
            raise ValueError("point_ids matching requires complete unique known identities")
        lookup={v:j for j,v in enumerate(b)}
        pairs=[(i,lookup[v]) for i,v in enumerate(a) if v in lookup]
    elif method=="linear_interpolation":
        domain=comparison.get("domain",[])
        if len(domain)!=2 or not finite(domain[0])<finite(domain[1]):
            raise ValueError("curve comparison requires explicit common domain")
        if min(len(rx),len(gx))<2 or any(y is None for y in [*ry,*gy]) or len(set(rx))!=len(rx) or len(set(gx))!=len(gx):
            raise ValueError("interpolation requires single-valued curves")
        if domain[0]<max(min(rx),min(gx)) or domain[1]>min(max(rx),max(gx)):
            raise ValueError("curve domain would require extrapolation")
        count=comparison.get("samples",101)
        if type(count) is not int or not 2<=count<=10000:
            raise ValueError("curve samples must be 2–10000")
        def sample(xs: list,ys: list,points: list) -> list:
            from bisect import bisect_right
            ordered=sorted(zip(xs,ys));xx,yy=zip(*ordered)
            values=[]
            for p in points:
                i=max(0,min(len(xx)-2,bisect_right(xx,p)-1))
                values.append(yy[i]+(yy[i+1]-yy[i])*(p-xx[i])/(xx[i+1]-xx[i]))
            return values
        grid=[domain[0]+i*(domain[1]-domain[0])/(count-1) for i in range(count)]
        ry,gy=sample(rx,ry,grid),sample(gx,gy,grid)
        rx=gx=grid;pairs=list(zip(range(count),range(count)))
        interpolated={"method":"piecewise_linear_in_physical_coordinates","domain":domain,"samples":count,
                      "point_detection_metrics_applicable":False}
    elif method=="unique_nearest":
        if len(rx)*len(gx)>2_000_000:
            return {"status":"not_comparable","reason":"matching resource limit; use explicit IDs or curve domain"}
        # Reciprocal unique nearest neighbors abstain on ties, avoiding row-order matches.
        edges=[]
        for i,(x,y) in enumerate(zip(rx,ry)):
            for j,(xx,yy) in enumerate(zip(gx,gy)):
                if y is None or yy is None:
                    if result.get("geometry")!=reference.get("geometry") or result.get("geometry")!="band_envelopes":continue
                    dy=0
                else:dy=abs(y-yy)/ty
                dx=(max(abs(a-b) for a,b in zip(result["bins"][i],reference["bins"][j]))/tx
                    if result.get("bins") else abs(x-xx)/tx)
                if max(dx,dy)<=1:edges.append((dx*dx+dy*dy,i,j))
        best_a={};best_b={}
        for score,i,j in sorted(edges):
            best_a.setdefault(i,[]).append((score,j));best_b.setdefault(j,[]).append((score,i))
        for i,candidates in best_a.items():
            score,j=candidates[0];reverse=best_b[j]
            tie=(len(candidates)>1 and math.isclose(score,candidates[1][0],abs_tol=1e-14)) or (len(reverse)>1 and math.isclose(reverse[0][0],reverse[1][0],abs_tol=1e-14))
            if tie:ambiguous+=1
            elif reverse[0][1]==i:pairs.append((i,j))
    else:
        raise ValueError("unsupported comparison method")
    absolute={"x":[],"y":[]};relative={"x":[],"y":[]};near_zero={"x":0,"y":0}
    endpoint={"lower":[],"upper":[]};bin_errors=[]
    correct=0
    for i,j in pairs:
        good=True
        for axis,a,b,tol in (("x",rx[i],gx[j],tx),("y",ry[i],gy[j],ty)):
            if a is None or b is None:continue
            error=abs(a-b);absolute[axis].append(error);good=good and error<=tol
            if abs(b)>1e-12:relative[axis].append(error/abs(b))
            else:near_zero[axis]+=1
        correct+=int(good)
        if interpolated is None:
            ea,eb=result.get("displayed_uncertainties",[]),reference.get("displayed_uncertainties",[])
            if ea and eb and ea[i].get("semantics")==eb[j].get("semantics"):
                for key in endpoint:
                    if ea[i].get(key) is not None and eb[j].get(key) is not None:
                        endpoint[key].append(abs(ea[i][key]-eb[j][key]))
            if result.get("bins") and reference.get("bins"):
                bin_errors.extend(abs(a-b) for a,b in zip(result["bins"][i],reference["bins"][j]))
    return {"status":"compared","method":method,"pairs":pairs,"ambiguous_detections":ambiguous,
            "matched":len(pairs),"reference_points":len(gx),"detected_points":len(rx),
            "correct_fraction":correct/len(gx),"false_detections":len(rx)-len(pairs) if interpolated is None else None,
            "missing_fraction":1-len(pairs)/len(gx) if interpolated is None else None,
            "matching_bin_fraction":len(pairs)/len(gx) if bin_errors else None,
            "missing_bin_fraction":1-len(pairs)/len(gx) if bin_errors else None,
            "absolute_errors":{k:_summary(v) for k,v in absolute.items()},
            "relative_errors":{k:_summary(v) for k,v in relative.items()},"near_zero_relative_omitted":near_zero,
            "near_zero_policy":"absolute error only when abs(reference)<=1e-12",
            "endpoint_errors":{k:_summary(v) for k,v in endpoint.items()},"bin_edge_errors":_summary(bin_errors),
            "interpolation":interpolated,"manual_review":"pending"}


def original_series(record: dict, series: dict, mapping: dict) -> dict:
    """Normalize only verified 1D originals; retain their explicit units and error conventions."""
    from copy import deepcopy

    result=deepcopy(series)
    bins=result.get("bins") or []
    if record["backend"]=="native":
        units={axis:record.get("axes",{}).get(axis,{}).get("unit") for axis in ("x","y")}
    else:
        independent=series.get("independent_variables",[])
        if len(independent)!=1:raise ValueError("benchmark requires a one-dimensional original series")
        units={"x":independent[0].get("header",{}).get("units"),"y":series.get("header",{}).get("units")}
        if bins and all(b.get("low") is not None and b.get("high") is not None for b in bins):
            result["bins"]=[[b["low"],b["high"]] for b in bins]
            # Used only to position bin comparisons; matching is on both edges, not inferred measurements.
            result["x"]=[(a+b)/2 for a,b in result["bins"]]
            result["coordinate_assumption"]="bin midpoint for matching/display only"
        elif any(b.get("low") is not None or b.get("high") is not None for b in bins):
            raise ValueError("incomplete original bin boundaries")
        else:result["bins"]=None
    if units!=mapping.get("units"):
        raise ValueError("mapping units differ from explicitly published/native units; no implicit conversion")
    result.update(series_id=mapping["series_id"],observable=mapping["observable"],units=units,
                  coordinate_definition=mapping["coordinate_definition"])
    components=series.get("uncertainties",[])
    result["displayed_uncertainties"]=[]
    if components and record["backend"]=="native":
        if len(components)!=1:raise ValueError("select an unambiguous native error component")
        error=components[0]
        for i,y in enumerate(result["y"]):
            scale=abs(y) if error.get("relative") else 1
            lower=error["symerror"][i] if error.get("representation")=="symmetric" else error["minus"][i]
            upper=error["symerror"][i] if error.get("representation")=="symmetric" else error["plus"][i]
            result["displayed_uncertainties"].append({"lower":y-abs(lower)*scale,"upper":y+abs(upper)*scale,
                                                     "semantics":"unlabeled_displayed_error"})
    elif components:
        selected=mapping.get("uncertainty_label")
        for row,y in zip(components,result["y"]):
            choices=[c for c in row["components"] if c.get("label")==selected]
            if len(choices)!=1:
                result["displayed_uncertainties"]=[]
                result["uncertainty_comparison"]="not_selected_or_ambiguous_component"
                break
            error=choices[0]
            def delta(component: dict) -> float:
                value=component.get("value")
                if value is None:raise ValueError("nonnumeric published error cannot be benchmarked")
                return finite(value)*(abs(y)/100 if component.get("relative") else 1)
            if "symerror" in error:
                upper=abs(delta(error["symerror"]));lower=-upper
            else:
                lower=delta(error["asymerror"]["minus"]);upper=delta(error["asymerror"]["plus"])
            result["displayed_uncertainties"].append({"lower":y+lower,"upper":y+upper,
                                                     "semantics":error.get("kind") or "unlabeled_displayed_error"})
    structural_check(result)
    return result
