import json
from pathlib import Path
from inertial_perception.simulation import export_demo,compare_modes

if __name__ == '__main__':
    out=export_demo('web/data/demo.json','combined',10.0,42)
    for name,data in out.items():
        print(f"{name:12s} orientation_RMSE={data['metrics']['orientation_rmse_deg']:.3f} deg")

    translation=compare_modes('translation',4.0,21,'ins_eskf')
    Path('web/data/translation_eval.json').write_text(json.dumps(translation,separators=(',',':')),encoding='utf-8')
    print('Translation evaluation:')
    for name,data in translation.items():
        m=data['metrics']
        print(f"{name:12s} position_RMSE={m['position_rmse_m']:.4f} m vertical_RMSE={m['vertical_position_rmse_m']:.4f} m velocity_RMSE={m['velocity_rmse_mps']:.4f} m/s")
