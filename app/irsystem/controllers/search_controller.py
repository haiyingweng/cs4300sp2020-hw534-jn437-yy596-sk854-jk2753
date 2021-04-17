from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from .calculate_similarity import *
project_name = "Cerealizer"
net_id = "Joie Ng: jn437, Ying Yang: yy596, Haiying Weng: hw534, Jason Jungwoo Kim: jk2753, Sooah Kang: sk854"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
		print(data)
	else:
		output_message = "Your search: " + query
		ranked_cereals = rank_by_similarity(query, inverted_index, idf, norms)
		data = get_cereal_details(ranked_cereals)
		if not data:
			data = ["None found"]
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



