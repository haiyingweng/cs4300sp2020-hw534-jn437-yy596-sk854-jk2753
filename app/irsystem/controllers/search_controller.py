from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from .calculate_similarity import *
from .db_related import *
from .similar_cereals import *

project_name = "Cerealizer"
net_id = "Joie Ng: jn437, Ying Yang: yy596, Haiying Weng: hw534, Jason Jungwoo Kim: jk2753, Sooah Kang: sk854"


@irsystem.route("/", methods=["GET"])
def search():
    search_by = request.args.get("search_by")
    if search_by == "cereal":
        query = request.args.get("cereal")
    else:
        query = request.args.get("keywords")
    filters = filteritems(request)
    if not query:
        query = ""
        data = []
    else:
        # ranked_cereals = rank_by_similarity(query, inverted_index, idf, norms, filters)
        if search_by == "cereal":
            ranked_cereals = rank_by_similar_cereal(query, cereal_sims_cos, filters)
        else:
            ranked_cereals = ranking_rocchio(query, tf_idf_matrix, filters)
        data = get_cereal_details(ranked_cereals)
    return render_template(
        "search.html",
        name=project_name,
        netid=net_id,
        search_by=search_by,
        query=query,
        data=data,
        all_cereals=all_cereals,
    )


@irsystem.route("/cerealname", methods=["GET"])
def click():
    name = request.args.get("cereal_name")
    query = request.args.get("query")
    tcin = request.args.get("tcin")
    score = request.args.get("score")
    search_by = request.args.get("search_by")
    # add cereal and keyword to database
    if search_by == "keyword":
        add_cereal_for_query(query, name)

    # get details for clicked cereal
    details = get_cereal_details([(tcin, float(score))])
    # print(details)

    # TODO: return page for the clicked cereal details
    # return render_template("____.html",target=name)
    return render_template(
        "search.html", name=name, score=score, details=details, query=query
    )
