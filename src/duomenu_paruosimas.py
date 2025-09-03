

def pasalinti_nereikalingas(df):
    stulpeliai_salinimui = ['Name', 'Cabin']
    df = df.drop(columns=stulpeliai_salinimui)
    return df



