"""Project's pre-existing VSI scoring protocol, isolated and explicitly named."""
from collections import defaultdict

from .legacy.vsi_acc import process_vsibench_single


class ProjectVSI:
    def __init__(self, adapter):
        self.adapter = adapter

    def prepare_dataset(self, rows):
        return self.adapter.prepare_dataset(rows)

    def evaluate_results(self, samples, outputs):
        results = []
        for sample, output in zip(samples, outputs):
            meta = sample["metadata"]
            item = {**meta, "ground_truth": sample["answer"], "pred_answer": output}
            bucket = process_vsibench_single("", "all", data=[item])
            task = next(iter(bucket))
            results.append({"id": meta.get("id"), "question_type": task,
                            "ground_truth": sample["answer"], "score": bucket[task]["scores"][0],
                            "empty_pred": bucket[task]["empty_pred"]})
        return results

    def compute_statistics(self, results):
        by_type = defaultdict(list)
        for row in results:
            by_type[row["question_type"]].append(row["score"])
        categories = {key: sum(values)/len(values) for key,values in by_type.items()}
        return {"overall_score": sum(row["score"] for row in results)/len(results),
                "category_results": categories, "macro_category_score": sum(categories.values())/len(categories),
                "total_samples": len(results), "empty_predictions": sum(row["empty_pred"] for row in results)}
