from sklearn.metrics import mean_absolute_error, make_scorer


def get_metric_name_mapping():
    return {_mae(): mean_absolute_error}


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _mae(): make_scorer(mean_absolute_error, greater_is_better=False, **params),
        _bne(): make_scorer(bike_number_error, understock_price=0.3, overstock_price=0.7)
    }
    return mapping[name]


def _mae():
    return "mean absolute error"

def _bne():
    return "bike number error"

def bike_number_error(y_true, y_pred, understock_price=0.7, overstock_price=0.3):
  e = (y_true - y_pred).astype(np.float32)
  factor = np.ones_like(e)
  factor[e > 0] = understock_price
  
  factor[e < 0] = overstock_price
  return np.sum(np.abs(e) * factor) / len(e)