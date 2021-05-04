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
    query = request.args.get("search")
    filters = filteritems(request)
    if not query:
        query = ""
        data = []
    else:
        output_message = "Your search: " + query
        # ranked_cereals = rank_by_similarity(query, inverted_index, idf, norms, filters)
        ranked_cereals = ranking_rocchio(query, tf_idf_matrix, filters)
        data = get_cereal_details(ranked_cereals)
    return render_template(
        "search.html",
        name=project_name,
        netid=net_id,
        query=query,
        data=data,
    )


@irsystem.route("/cerealname", methods=["GET"])
def click():
    name = request.args.get("cereal_name")
    query = request.args.get("query")
    tcin = request.args.get("tcin")
    score = request.args.get("score")
    # add cereal and keyword to database
    add_cereal_for_query(query, name)

    # get details for clicked cereal
    details = get_cereal_details([(tcin, float(score))])
    # print(details)

    # TODO: return page for the clicked cereal details
    # return render_template("____.html",target=name)
    return render_template(
        "search.html", name=name, score=score, details=details, query=query
    )
