"""Offline visual fixtures; optional installed-engine tests are identified separately."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kamiknows.common import read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_manifest
from kamiknows.kamiplotextractor.digitization import (
    Axis, Backend, Calibration, ExtractionError, import_table, raster_series, vector_series)
from kamiknows.kamiplotextractor.records import PlotDataRecord
from kamiknows.kamiplotextractor import run_benchmark, run_classification, run_digitization, run_inventory
from kamiknows.kamiplotextractor.validation import compare_series


def config(system="pdf_points"):
    return {"figure_id": "fig", "panel_id": None, "asset_id": "asset", "page": 1,
            "roi": [0, 0, 100, 100], "coordinate_system": system,
            "calibration_source": "axis_ticks", "manual_interventions": ["selected axes and paths"],
            "x": {"scale": "linear", "anchors": [{"position": 0, "value": 0}, {"position": 100, "value": 10}], "unit": "GeV"},
            "y": {"scale": "linear", "anchors": [{"position": 0, "value": 10}, {"position": 100, "value": 0}], "unit": "events"},
            "series": [{"series_id": "s", "observable": "yield", "coordinate_definition": "x energy, y yield",
                        "geometry": "line", "path_indices": [0]}]}


def line(points):
    return {"items": [["l", a, b] for a,b in zip(points, points[1:])], "rect": [0,0,100,100], "closePath": False}


def rectangle(box):
    return {"items": [["re", box, 1]], "rect": box, "closePath": True}


def test_schema_preserves_12():
    r = PlotDataRecord(None,None,None,"d","unavailable",{},{},{})
    assert "calibration" not in r.to_dict()
    r.schema_version = "1.3"
    assert r.to_dict()["calibration"] == {}


def test_calibration_linear_log_inversion_and_residuals():
    c = config()
    c["x"]["anchors"].append({"position": 50,"value": 5})
    c["y"].update(scale="log10",anchors=[{"position": 0,"value": 10000},{"position": 100,"value": 1}])
    cal = Calibration(c)
    assert cal.convert([[50,50]]) == ([5],[100])
    assert cal.y.position(100) == 50
    assert cal.metadata()["fit"]["x"]["residuals"] == [0,0,0]


@pytest.mark.parametrize("mutation", [
    lambda c:c["x"].update(anchors=[{"position":0,"value":0}]),
    lambda c:c["x"]["anchors"][1].update(position=0),
    lambda c:c["x"]["anchors"][1].update(value=0),
    lambda c:c["x"].update(scale="log10"),
    lambda c:c["x"]["anchors"][1].update(position=101),
    lambda c:c["x"]["anchors"][1].update(value=float("nan")),
    lambda c:c["y"].update(direction="increasing"),
    lambda c:c["x"]["anchors"].append({"position":50,"value":6}),
    lambda c:c.update(page=0),
    lambda c:c.update(roi=[0,0,0,100]),
])
def test_bad_calibration(mutation):
    c=config(); mutation(c)
    with pytest.raises(ValueError):Calibration(c)


def test_vector_line_multiseries_and_bezier():
    c=config()
    c["series"].append({**c["series"][0],"series_id":"t","path_indices":[1]})
    bez={"items":[["c",[0,100],[20,80],[80,20],[100,0]]],"rect":[0,0,100,100]}
    result=vector_series([line([[10,90],[50,50],[90,10]]),bez],Calibration(c))
    assert result[0]["x"] == [1,5,9] and result[0]["y"] == [1,5,9]
    assert len(result[1]["x"]) == 17
    assert result[1]["sampling"]["original_measurement_points"] is False


def test_scatter_log_errors_and_histogram_band():
    c=config();c["series"][0].update(geometry="scatter",path_indices=[0],error_path_indices=[1])
    c["y"].update(scale="log10",anchors=[{"position":0,"value":10000},{"position":100,"value":1}])
    result=vector_series([rectangle([48,48,52,52]),line([[50,25],[50,75]])],Calibration(c))[0]
    assert result["y"] == [100]
    assert result["displayed_uncertainties"][0]["lower"] == 10
    assert result["displayed_uncertainties"][0]["upper"] == 1000
    c=config();c["series"][0].update(geometry="histogram",path_indices=[0,1],baseline_position=100)
    result=vector_series([rectangle([0,50,20,100]),rectangle([20,20,40,100])],Calibration(c))[0]
    assert result["bins"] == [[0,2],[2,4]] and result["y"] == [5,8]
    c["series"][0].update(geometry="band")
    result=vector_series([line([[10,20],[50,30]]),line([[10,40],[50,50]])],Calibration(c))[0]
    assert result["y"] == [None,None]
    assert result["displayed_uncertainties"][0]["lower"] == 6


def test_ambiguous_hidden_stacked_and_outside():
    c=config(); c["series"][0].update(geometry="scatter",path_indices=[0,1])
    with pytest.raises(ExtractionError,match="overlapping"):
        vector_series([rectangle([10,10,16,16]),rectangle([12,12,18,18])],Calibration(c))
    c["series"][0].update(stacked=True)
    with pytest.raises(ExtractionError,match="stacked"):vector_series([rectangle([0,0,10,10])]*2,Calibration(c))
    c=config()
    with pytest.raises(ExtractionError,match="out-of-ROI"):vector_series([line([[0,0],[101,0]])],Calibration(c))
    c["series"]=[]
    with pytest.raises(ExtractionError,match="path_indices"):vector_series([],Calibration(c))


def test_capabilities_and_missing_dependency(monkeypatch,tmp_path):
    monkeypatch.setattr("importlib.util.find_spec",lambda name:None)
    assert Backend("plotsift").capabilities()["execution"] == "import_only"
    assert Backend("plotdigitizer").capabilities()["multiple_series"] is False
    with pytest.raises(ExtractionError,match="missing"):
        Backend("pymupdf").digitize(tmp_path/"a.pdf",Calibration(config()),tmp_path)


def test_raster_batch_coordinates_and_immutable_source(monkeypatch,tmp_path):
    from PIL import Image
    source=tmp_path/"plot.png";Image.new("RGB",(100,100),"white").save(source)
    before=source.read_bytes();c=config("pixels");c["single_curve_evidence"]="synthetic single line"
    def fake(command,directory,stdout,stderr):
        assert command[1].endswith("preprocessed.png") and command.count("-p")==command.count("-l")==3
        Path(command[-1]).write_text("10 10\n50 50\n90 90\n")
    monkeypatch.setattr("kamiknows.kamiplotextractor.digitization.run_bounded",fake)
    data,meta=raster_series(source,Calibration(c),tmp_path)
    assert data[0]["x"]==data[0]["y"]==[1,5,9]
    assert source.read_bytes()==before and meta["transforms"][1]["operation"]=="grayscale"
    c.pop("single_curve_evidence")
    with pytest.raises(ExtractionError,match="evidence"):raster_series(source,Calibration(c),tmp_path)


def test_subprocess_timeout(monkeypatch,tmp_path):
    from kamiknows.kamiplotextractor.digitization import run_bounded
    class Process:
        killed=False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def poll(self):return None
        def kill(self):self.killed=True
        def wait(self):return 0
    process=Process()
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**kw:process)
    times=iter([0,46]);monkeypatch.setattr("time.monotonic",lambda:next(times))
    with pytest.raises(ExtractionError,match="timeout"):
        run_bounded(["plotdigitizer"],tmp_path,None,None)
    assert process.killed


@pytest.mark.parametrize("backend,suffix,delimiter,content",[
    ("chartdetective","csv",",","X,Y\n1,2\n3,4\n"),
    ("plotsift","csv",",","x,y,y_error\n1,2,0.5\n3,4,\n"),
    ("plotsift","tsv","\t","x\ty\ty_error\n1\t2\t0.5\n3\t4\t\n"),
    ("plotsift","dat","space","# x y y_error\n1 2 0.5\n3 4\n"),
])
def test_import_formats_and_semantics(tmp_path,backend,suffix,delimiter,content):
    path=tmp_path/f"points.{suffix}";path.write_text(content)
    entry={"backend":backend,"backend_version":"test-commit","file":path.name,"export_sha256":sha256_file(path),
           "delimiter":delimiter,"columns":{"x":0,"y":1,"y_error":2},"series_id":"s"}
    data,_=import_table(tmp_path,entry,Calibration(config("pixels")))
    assert data["x"]==[1,3] and data["y"]==[2,4]
    if backend=="plotsift":
        assert data["displayed_uncertainties"][0]["magnitude"]==0.5
        assert data["displayed_uncertainties"][0]["lower"] is None
        assert data["displayed_uncertainties"][0]["semantics"]=="unlabeled_displayed_error"
    entry["export_sha256"]="wrong"
    with pytest.raises(ValueError,match="SHA256"):import_table(tmp_path,entry,Calibration(config()))
    entry["file"]="../escape.csv"
    with pytest.raises(ValueError,match="contained"):import_table(tmp_path,entry,Calibration(config()))


@pytest.fixture
def visual_run(tmp_path):
    """Known pixels, parsed source, inventory and classification with real manifests."""
    from PIL import Image, ImageDraw
    from kamiknows import parsing
    parsed=tmp_path/"parsed";root=parsed/"papers"/"synthetic"/"source"
    root.mkdir(parents=True)
    image=Image.new("RGB",(100,100),"white");ImageDraw.Draw(image).line([(10,90),(90,10)],fill="black",width=2)
    image.save(root/"plot.png")
    latex=(r"\documentclass{article}\begin{document}\section{Results}\begin{figure}\includegraphics{plot.png}"
           r"\begin{axis}[x unit={GeV},y unit={events}]\addplot coordinates {(1,1) (5,5) (9,9)};\end{axis}"
           r"\caption{Energy distribution plot.}\end{figure}\end{document}")
    (root/"main.tex").write_text(latex)
    flat=parsing.flatten_latex_file(root/"main.tex",root)
    (root.parent/"flat.tex").write_text(flat)
    sections,_,_=parsing.parse_latex(flat);write_json(root.parent/"sections.json",sections)
    write_jsonl(parsed/"papers.jsonl",[{"paper_id":"synthetic","status":"success"}])
    write_manifest(parsed,step="parse",params={},inputs={},summary={})
    inv=tmp_path/"inventory";classified=tmp_path/"classified"
    run_inventory(parsed_dir=parsed,output_dir=inv);run_classification(inventory_dir=inv,output_dir=classified)
    figure=read_jsonl(classified/"figures.jsonl")[0];asset=figure["source_assets"][0]
    c=config("pixels");c.update(figure_id=figure["figure_id"],asset_id=asset["asset_id"],single_curve_evidence="known synthetic line")
    calibration=tmp_path/"calibration.json";write_json(calibration,{"schema_version":"1","plots":[c]})
    return parsed,classified,figure,asset,c,calibration


def install_fake_raster(monkeypatch):
    from kamiknows.kamiplotextractor.digitization import series_record
    def fake(self,path,calibration,work_dir,dpi=144):
        return [series_record(calibration.config["series"][0],[[10,90],[50,50],[90,10]],calibration,"marker_centers")],{"render_scale":1}
    monkeypatch.setattr(Backend,"digitize",fake)


def test_pipeline_compare_benchmark_and_immutability(visual_run,tmp_path,monkeypatch):
    parsed,classified,figure,asset,c,calibration=visual_run
    before={p:p.read_bytes() for base in (parsed,classified,tmp_path/"inventory") for p in base.rglob("*") if p.is_file()}
    install_fake_raster(monkeypatch)
    output=tmp_path/"digitized"
    summary=run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=output,
                             backend="compare",calibration_file=calibration)
    records=read_jsonl(output/"digitized_data.jsonl")
    assert summary["records"]==4 and summary["extraction_status_counts"]["success"]==1
    assert len(summary["comparisons"])==6
    success=next(r for r in records if r["extraction_status"]=="success")
    assert success["schema_version"]=="1.3" and success["source_method"]=="raster_digitized"
    assert (output/"overlays"/success["data_id"]/"overlay.svg").is_file()
    reference={"series_id":"s","observable":"yield","coordinate_definition":"x energy, y yield",
               "units":{"x":"GeV","y":"events"},"x":[9,5,1],"y":[9,5,1]}
    truth=tmp_path/"truth.json"
    write_json(truth,{"schema_version":"1","references":[{"figure_id":figure["figure_id"],"asset_id":asset["asset_id"],"panel_id":None,
        "source_kind":"synthetic","independent":True,"provenance":{"generator":"exact line y=x, axis ticks independent"},
        "series":reference,"comparison":{"tolerance":{"x":.01,"y":.01}}}]})
    results=run_benchmark(digitized_dir=output,output_dir=tmp_path/"benchmark",ground_truth_file=truth)
    assert results["coverage"]==.25 and results["abstention_rate"]==.75 and results["mean_correct_fraction"]==1
    detail=next(d for d in read_jsonl(tmp_path/"benchmark"/"benchmark_details.jsonl") if d["status"]=="compared")
    assert detail["pairs"]==[[0,2],[1,1],[2,0]]
    assert detail["absolute_errors"]["y"]["p95"]==0
    assert all(p.read_bytes()==v for p,v in before.items())
    with pytest.raises((ValueError,FileExistsError)):run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=output)


def export_fixture(tmp_path,figure,asset,c):
    imports=tmp_path/"imports";imports.mkdir()
    data=imports/"sheet.csv";data.write_text("x,y,y_error\n1,1,0.2\n5,5,\n9,9,0.3\n")
    e={"paper_id":figure["paper_id"],"figure_id":figure["figure_id"],"panel_id":None,"asset_id":asset["asset_id"],
       "asset_sha256":asset["sha256"],"backend":"plotsift","backend_version":"offline-fixture", "file":"sheet.csv",
       "export_sha256":sha256_file(data),"calibration":c,"series_id":"s","sheet":"signal",
       "columns":{"x":0,"y":1,"y_error":2},"observable":"yield","coordinate_definition":"x energy, y yield"}
    write_json(imports/"imports.json",{"schema_version":"1","exports":[e]})
    return imports,e


def test_import_auto_mapping_and_hashes(visual_run,tmp_path):
    parsed,classified,figure,asset,c,calibration=visual_run
    imports,e=export_fixture(tmp_path,figure,asset,c)
    output=tmp_path/"auto"
    summary=run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=output,imports_dir=imports)
    assert summary["extraction_status_counts"]=={"success":1}
    assert read_jsonl(output/"digitized_data.jsonl")[0]["source_method"]=="manual_digitized"
    e["asset_sha256"]="bad";write_json(imports/"imports.json",{"schema_version":"1","exports":[e]})
    summary=run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=tmp_path/"badsha",backend="plotsift",imports_dir=imports)
    assert summary["extraction_status_counts"]=={"backend_error":1}
    e["figure_id"]="other";write_json(imports/"imports.json",{"schema_version":"1","exports":[e]})
    with pytest.raises(ValueError,match="mapping"):
        run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=tmp_path/"badmap",backend="plotsift",imports_dir=imports)


def test_missing_calibration_source_change_and_selection(visual_run,tmp_path,monkeypatch):
    parsed,classified,figure,asset,c,calibration=visual_run
    summary=run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=tmp_path/"missing",backend="plotdigitizer")
    assert summary["extraction_status_counts"]=={"needs_calibration":1}
    (parsed/"papers"/"synthetic"/"source"/"plot.png").write_bytes(b"changed")
    summary=run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=tmp_path/"changed",calibration_file=calibration)
    assert summary["extraction_status_counts"]=={"source_mismatch":1}


def test_original_skip_override_and_recovery_benchmark(visual_run,tmp_path,monkeypatch):
    from kamiknows.kamiplotextractor import run_recovery
    parsed,classified,figure,asset,c,calibration=visual_run
    recovery=tmp_path/"recovery"
    run_recovery(parsed_dir=parsed,classified_dir=classified,output_dir=recovery)
    original=next(r for r in read_jsonl(recovery/"plot_data.jsonl") if r["source_method"]=="original_source_data")
    install_fake_raster(monkeypatch)
    kwargs=dict(parsed_dir=parsed,classified_dir=classified,backend="plotdigitizer",calibration_file=calibration,recovery_dir=recovery)
    # Figure-level data alone does not prove that this included PNG represents it.
    assert run_digitization(**kwargs,output_dir=tmp_path/"unmapped")["extraction_status_counts"]=={"success":1}
    c["original_data_ids"]=[original["data_id"]]
    c["benchmark_references"]=[{"data_id":original["data_id"],"source_series_id":original["series"][0]["series_id"],
        "series_id":"s","observable":"yield","units":{"x":"GeV","y":"events"},"coordinate_definition":"x energy, y yield",
        "comparison":{"tolerance":{"x":.01,"y":.01}}}]
    write_json(calibration,{"schema_version":"1","plots":[c]})
    assert run_digitization(**kwargs,output_dir=tmp_path/"skipped")["extraction_status_counts"]=={"skipped_original":1}
    output=tmp_path/"override"
    assert run_digitization(**kwargs,output_dir=output,selection="all")["extraction_status_counts"]=={"success":1}
    benchmark=run_benchmark(digitized_dir=output,recovery_dir=recovery,output_dir=tmp_path/"native_benchmark")
    assert benchmark["mean_correct_fraction"]==1 and benchmark["coverage"]==1


def test_original_hepdata_bins_errors_and_unit_mismatch():
    from kamiknows.kamiplotextractor.source_data import parse_hepdata_table
    from kamiknows.kamiplotextractor.validation import original_series
    table={"independent_variables":[{"header":{"name":"Energy","units":"GeV"},"values":[{"low":1,"high":3}]}],
           "dependent_variables":[{"header":{"name":"yield","units":"events"},"values":[{"value":10,"errors":[{"label":"stat","symerror":"10%"}]}]}]}
    axes,series,_=parse_hepdata_table(table)
    mapping={"series_id":"s","observable":"yield","coordinate_definition":"xy","units":{"x":"GeV","y":"events"},"uncertainty_label":"stat"}
    result=original_series({"backend":"hepdata","axes":axes},series[0],mapping)
    assert result["bins"]==[[1,3]] and result["x"]==[2]
    assert result["displayed_uncertainties"]==[{"lower":9,"upper":11,"semantics":"statistical"}]
    with pytest.raises(ValueError,match="units"):
        original_series({"backend":"hepdata","axes":axes},series[0],{**mapping,"units":{"x":"MeV","y":"events"}})


def test_benchmark_leakage_and_invalid_reference(visual_run,tmp_path,monkeypatch):
    parsed,classified,figure,asset,c,calibration=visual_run
    install_fake_raster(monkeypatch)
    c["calibration_source"]="evaluation_points"
    write_json(calibration,{"schema_version":"1","plots":[c]})
    output=tmp_path/"digitized"
    run_digitization(parsed_dir=parsed,classified_dir=classified,output_dir=output,backend="plotdigitizer",calibration_file=calibration)
    reference={"figure_id":figure["figure_id"],"asset_id":asset["asset_id"],"panel_id":None,"source_kind":"synthetic",
        "independent":True,"provenance":{"generator":"known y=x"},"series":{"series_id":"s","x":[1,5,9],"y":[1,5,9],
        "observable":"yield","coordinate_definition":"x energy, y yield","units":{"x":"GeV","y":"events"}},
        "comparison":{"tolerance":{"x":.01,"y":.01}}}
    truth=tmp_path/"truth.json";write_json(truth,{"schema_version":"1","references":[reference]})
    result=run_benchmark(digitized_dir=output,ground_truth_file=truth,output_dir=tmp_path/"benchmark")
    assert result["compared"]==0 and result["abstention_rate"]==1
    reference["source_kind"]="raster_digitized";write_json(truth,{"schema_version":"1","references":[reference]})
    with pytest.raises(ValueError,match="independent"):
        run_benchmark(digitized_dir=output,ground_truth_file=truth,output_dir=tmp_path/"invalid")


def test_bin_edge_matching_and_endpoint_metrics():
    s={"series_id":"s","observable":"yield","coordinate_definition":"bins","units":{"x":"","y":""},
       "x":[1,3],"y":[5,7],"bins":[[0,2],[2,4]],"displayed_uncertainties":[
           {"lower":4,"upper":6,"semantics":"unlabeled_displayed_error"},{"lower":6,"upper":8,"semantics":"unlabeled_displayed_error"}]}
    t=copy.deepcopy(s);t["bins"][0]=[-1,3]
    result=compare_series(s,t,{"tolerance":{"x":.1,"y":.1}})
    assert result["matching_bin_fraction"]==.5 and result["missing_bin_fraction"]==.5
    assert result["endpoint_errors"]["lower"]["max"]==0


def test_metrics_false_missing_nearzero_and_interpolation():
    a={"series_id":"s","observable":"yield","coordinate_definition":"xy","units":{"x":"","y":""},
       "x":[0,1,9],"y":[0,1.1,9]}
    b={**a,"x":[0,1,2],"y":[0,1,2]}
    result=compare_series(a,b,{"tolerance":{"x":.2,"y":.2}})
    assert result["correct_fraction"]==2/3 and result["false_detections"]==1
    assert result["near_zero_relative_omitted"]=={"x":1,"y":1}
    result=compare_series({**a,"x":[0,2],"y":[0,2]},b,
                          {"method":"linear_interpolation","domain":[0,2],"samples":9,"tolerance":{"x":.1,"y":.1}})
    assert result["absolute_errors"]["y"]["max"]==0 and result["false_detections"] is None
    assert compare_series(a,{**b,"observable":"other"},{})["status"]=="not_comparable"


def test_cli_modes_and_strict_options(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts"/"extract_plots.py"
    for arguments in (["--mode","inventory","--parsed-dir","unused","--backend","auto"],
                      ["--mode","benchmark","--digitized-dir","unused"],
                      ["--mode","digitize","--classified-dir","unused","--parsed-dir","unused","--raster-dpi","500"]):
        completed=subprocess.run([sys.executable,str(script),"--output-dir",str(tmp_path/"out"),*arguments],capture_output=True,text=True)
        assert completed.returncode==2


def test_optional_pymupdf_integration(tmp_path):
    pymupdf=pytest.importorskip("pymupdf",reason="optional real PyMuPDF not installed")
    path=tmp_path/"source.pdf"
    with pymupdf.open() as doc:
        page=doc.new_page(width=100,height=100)
        shape=page.new_shape();shape.draw_polyline([(10,90),(50,50),(90,10)]);shape.finish(color=(0,0,0),closePath=False);shape.commit()
        doc.save(path)
    before=path.read_bytes()
    series,metadata=Backend("pymupdf").digitize(path,Calibration(config()),tmp_path)
    assert series[0]["x"]==[1,5,9] and series[0]["y"]==[1,5,9]
    assert metadata["page"]==1 and path.read_bytes()==before


def test_optional_plotdigitizer_integration(tmp_path):
    pytest.importorskip("plotdigitizer",reason="optional real plotdigitizer not installed")
    if not Backend("plotdigitizer").available():pytest.skip("optional plotdigitizer executable unavailable")
    from PIL import Image,ImageDraw
    source=tmp_path/"curve.png";image=Image.new("RGB",(100,100),"white")
    ImageDraw.Draw(image).line([(10,90),(90,10)],fill="black",width=2);image.save(source)
    c=config("pixels");c["single_curve_evidence"]="known synthetic diagonal"
    series,_=Backend("plotdigitizer").digitize(source,Calibration(c),tmp_path)
    assert len(series[0]["x"])>=70
    assert max(abs(x-y) for x,y in zip(series[0]["x"],series[0]["y"]))<.21
