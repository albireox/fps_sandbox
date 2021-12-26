from __future__ import annotations

import sys

import pandas
from pymodbus.client.sync import ModbusTcpClient


def get_chiller_data(variables_file: str):

    variables = pandas.read_csv(variables_file)

    data = []

    for ii, address in enumerate(variables.Address):
        client = ModbusTcpClient("10.25.1.162", 1111)
        register = client.read_holding_registers(address - 1).registers[0]
        register *= variables.Scale.iloc[ii]
        data.append(
            {
                "name": variables.Name.iloc[ii],
                "value": register,
                "read_only": variables.ReadOnly.iloc[ii],
                "description": variables.Description.iloc[ii],
            }
        )

    results = pandas.DataFrame.from_records(data)
    results.to_csv("../results/chiller_data.csv")


if __name__ == "__main__":
    get_chiller_data(sys.argv[1])
