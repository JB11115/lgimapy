from lgimapy.bloomberg import bdh

start = '3/1/2021'
end = None
spx = bdh('SPX', 'Index', 'PX_LAST', start, end)
other = bdh('RSX US', 'Equity', 'PX_LAST', start, end)
other
