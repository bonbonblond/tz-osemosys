from typing import Dict

import xarray as xr
from linopy import LinearExpression, Model


def add_commodity_balance_constraints(
    ds: xr.Dataset, m: Model, lex: Dict[str, LinearExpression]
) -> Model:
    """Add Commodity Balance constraints to the model.
    Ensures that production of gen-ELC equals production of final-ELC.
    
    This prevents overgeneration by forcing all generated electricity 
    to be transmitted/used exactly, with no excess allowed.

    Arguments
    ---------
    ds: xarray.Dataset
        The parameters dataset
    m: linopy.Model
        A linopy model
    lex: Dict[str, LinearExpression]
        A dictionary of linear expressions, persisted after solve

    Returns
    -------
    linopy.Model
    """
    
    # Get the Production linear expression which has dimensions [REGION, TIMESLICE, FUEL, YEAR]
    production = lex["Production"]
    
    if "gen-ELC" in ds["FUEL"].values and "final-ELC" in ds["FUEL"].values:
        # Select production for gen-ELC and final-ELC
        gen_elc_production = production.sel(FUEL="gen-ELC")
        final_elc_production = production.sel(FUEL="final-ELC")
        
        # Add constraint: gen-ELC production == final-ELC production totalised over all regions for each timeslice and year
        gen_elc_total = gen_elc_production.sum("REGION")
        final_elc_total = final_elc_production.sum("REGION")
        con = gen_elc_total == final_elc_total
        m.add_constraints(con, name="CommodityBalance_GenELC_Equals_FinalELC")

    if "primary-electricity" in ds["FUEL"].values and "secondary-electricity" in ds["FUEL"].values:
        # Select production for primary-electricity and secondary-electricity
        primary_elc_production = production.sel(FUEL="primary-electricity")
        secondary_elc_production = production.sel(FUEL="secondary-electricity")
        primary_elc_production = primary_elc_production.sum("REGION")
        secondary_elc_production = secondary_elc_production.sum("REGION")
        
        # Add constraint: primary-electricity production == secondary-electricity production totalised over all regions for each timeslice and year
        # con = primary_elc_production == secondary_elc_production
        # m.add_constraints(con, name="CommodityBalance_PrimarySecondaryElectricity")
        difference = primary_elc_production - secondary_elc_production
        m.add_constraints(difference <= 1e-3, name="CommodityBalance_PrimarySecondaryElectricity_upper")
        m.add_constraints(difference >= -1e-3, name="CommodityBalance_PrimarySecondaryElectricity_lower")
    return m
