def IG_sectors(with_tildes=False):
    """
    List[str]:
        IG sectors used for daily snapshot.
        Overlap exists with subsectors and sectors.
    """
    sectors = [
        "INDUSTRIALS",  # Industrials
        "BASICS",
        "~CHEMICALS",
        "~METALS_AND_MINING",
        "CAPITAL_GOODS",
        "COMMUNICATIONS",
        "~CABLE_SATELLITE",
        "~MEDIA_ENTERTAINMENT",
        "~WIRELINES_WIRELESS",
        "CONSUMER_CYCLICAL",
        "~AUTOMOTIVE",
        "~RETAILERS",
        "CONSUMER_NON_CYCLICAL",
        "~FOOD_AND_BEVERAGE",
        "~HEALTHCARE_EX_MANAGED_CARE",
        "~MANAGED_CARE",
        "~PHARMACEUTICALS",
        "~TOBACCO",
        "ENERGY",
        "~INDEPENDENT",
        "~INTEGRATED",
        "~OIL_FIELD_SERVICES",
        "~REFINING",
        "~MIDSTREAM",
        "ENVIRONMENTAL_IND_OTHER",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "~RAILROADS",
        "FINANCIALS",  # Financials
        "BANKS",
        "~SIFI_BANKS_SR",
        "~SIFI_BANKS_SUB",
        "~US_REGIONAL_BANKS",
        "~YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "LIFE",
        "P&C",
        "REITS",
        "UTILITY",  # Utilities
        "NON_CORP",  # Non-Corp
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
    ]
    if not with_tildes:
        sectors = [s.strip('~') for s in sectors]
    return sectors


def HY_sectors(with_tildes=False, sectors_only=True):
    """
    List[str]:
        HY Sectors used for daily snapshot. Not fully inclusive
        and overlap exists with subsectors and sectors.

        Missing sectors include: [
            'AUTO_LOANS',
            'DEPARTMENT_STORES',
            'FOOD_AND_DRUG_RETAILERS',
            'RESTAURANTS',
            'SPECIALTY_RETAIL',
            'THEATERS_AND_ENTERTAINMENT',
            'TOBACCO'
        ]
    """
    sectors = [
        "H4UN",
        "~BB",
        "~B",
        "HUC3_CCC",
        "LDB1_BBB",
        "AUTOMOTIVE",
        "~AUTOMAKERS",
        "~AUTOPARTS",
        "BASICS",
        "~HOME_BUILDERS",
        "~BUILDING_MATERIALS",
        "~CHEMICALS",
        "~METALS_AND_MINING",
        "CAPITAL_GOODS",
        "~AEROSPACE_DEFENSE",
        "~DIVERSIFIED_CAPITAL_GOODS",
        "~PACKAGING",
        "BEVERAGE",
        "FOOD",
        "PERSONAL_AND_HOUSEHOLD_PRODUCTS",
        "ENERGY",
        "~ENERGY_EXPLORATION_AND_PRODUCTION",
        "~GAS_DISTRIBUTION",
        "~OIL_REFINING_AND_MARKETING",
        "HEALTHCARE",
        "~HEALTH_FACILITIES",
        "~MANAGED_CARE",
        "~PHARMA",
        "GAMING",
        "HOTELS",
        "RECREATION_AND_TRAVEL",
        "REAL_ESTATE",
        "~REITS",
        "ENVIRONMENTAL",
        "SUPPORT_SERVICES",
        "TMT",
        "~CABLE_SATELLITE",
        "~MEDIA_CONTENT",
        "~TELECOM_SATELLITE",
        "~TELECOM_WIRELESS",
        "TECHNOLOGY",
        "~SOFTWARE",
        "~HARDWARE",
        "TRANSPORTATION",
        "~AIRLINES",
        "UTILITY",
    ]
    if not with_tildes:
        sectors = [s.strip('~') for s in sectors]
    if sectors_only:
        sectors = sectors[5:]
    return sectors

def credit_sectors():
    """
    List[str]:
        General sectors used for both IG and HY. Fully inclusive
        of both indices (though Sovs do not exist in HY) and no
        overlap.
    """
    return [
        "CHEMICALS",
        "CAPITAL_GOODS",
        "METALS_AND_MINING",
        "COMMUNICATIONS",
        "AUTOMOTIVE",
        "RETAILERS",
        "CONSUMER_CYCLICAL_EX_AUTOS_RETAILERS",
        "HEALTHCARE_PHARMA",
        "FOOD_AND_BEVERAGE",
        "CONSUMER_NON_CYCLICAL_OTHER",
        "ENERGY",
        "TECHNOLOGY",
        "TRANSPORTATION",
        "BANKS",
        "OTHER_FIN",
        "INSURANCE",
        "REITS",
        "UTILITY",
        "SOVEREIGN",
        "NON_CORP_OTHER",
        "OTHER_INDUSTRIAL",
    ]

def IG_market_segments():
    """
    List[str]:
        Market segments by rating, top level sector,
        maturity, and dollar price.
    """
    return [
        "AAA",
        "AA",
        "A",
        "A_FIN",
        "A_NON_FIN_TOP_30_10+_HiPX_LT25Y",
        "A_NON_FIN_TOP_30_10+_LowPX_LT25Y",
        "A_NON_FIN_TOP_30_10+_HiPX_GT25Y",
        "A_NON_FIN_TOP_30_10+_LowPX_GT25Y",
        "A_NON_FIN_EX_TOP_30_10+_HiPX_LT25Y",
        "A_NON_FIN_EX_TOP_30_10+_HiPX_GT25Y",
        "A_NON_FIN_EX_TOP_30_10+_LowPX_LT25Y",
        "A_NON_FIN_EX_TOP_30_10+_LowPX_GT25Y",
        "A_NON_CORP",
        "BBB",
        "BBB_FIN",
        "BBB_NON_FIN_TOP_30_10+_HiPX_LT25Y",
        "BBB_NON_FIN_TOP_30_10+_LowPX_LT25Y",
        "BBB_NON_FIN_TOP_30_10+_HiPX_GT25Y",
        "BBB_NON_FIN_TOP_30_10+_LowPX_GT25Y",
        "BBB_NON_FIN_EX_TOP_30_10+_HiPX_LT25Y",
        "BBB_NON_FIN_EX_TOP_30_10+_HiPX_GT25Y",
        "BBB_NON_FIN_EX_TOP_30_10+_LowPX_LT25Y",
        "BBB_NON_FIN_EX_TOP_30_10+_LowPX_GT25Y",
        "BBB_NON_CORP",
    ]
