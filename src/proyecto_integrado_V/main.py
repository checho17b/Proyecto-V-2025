import os
import pandas as pd
from modeller import Modeler
from collector import Collector
from logger import Logger

def main():
    logger = Logger()
    collector = Collector(logger)
    modeler = Modeler(logger)

    df = collector.collector_data()
    
    modeler.entrenar(df)
    
    preds = modeler.predecir(df)
    
    df['prediccion'] = preds

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, 'static', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'Apple_data.csv')
    df.to_csv(csv_path, index=False)
    
    logger.info('main', 'main', f'DataFrame con predicciones guardado en {csv_path}')
    print(f'DataFrame guardado en {csv_path}')

    print(df.head(20))

if __name__ == '__main__':
    main()
