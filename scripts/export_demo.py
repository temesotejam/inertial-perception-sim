from inertial_perception.simulation import export_demo

if __name__ == '__main__':
    out=export_demo('web/data/demo.json','combined',10.0,42)
    for name,data in out.items(): print(f"{name:12s} RMSE={data['metrics']['orientation_rmse_deg']:.3f} deg")
