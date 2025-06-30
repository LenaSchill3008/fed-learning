# fed-learning · Flower ⚘ + scikit-learn

Federated-learning demo on two tabular datasets (**Iris** and **Adult
Income**) using [Flower](https://flower.ai).

---

## Install

```bash
git clone <repo>
cd fed-learning
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e .
```

---

## One simulation

```bash
flwr run .                                # defaults to Iris
flwr run . --run-config 'dataset="adult"' # run Adult
```

Tune with `num-server-rounds`, `local-epochs`, `penalty`, etc.

---

## Batch run + CSV

`run.sh` launches both datasets, grabs **round-5** loss/accuracy from each
log and appends to `results/results.csv`.

```bash
chmod +x run.sh   # once
./run.sh
```

All logs live in `results/*.log`.

---

## Resources

* Docs  <https://flower.ai/docs>
* GitHub ⭐ <https://github.com/adap/flower>
* Community [Slack](https://flower.ai/join-slack/) | [Discuss](https://discuss.flower.ai/)
