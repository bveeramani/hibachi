from lime.lime_text import LimeTextExplainer
FEATURE_NAMES = ("a", "b", "c", "d", "e", "f", "g", "h")
CLASS_NAMES = ("non-functional", "functional")
explainer = LimeTabularExplainer(train, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES, categorical_features=[i for i in range(7)])
explanation = explainer.explain_instance(___, model, num_features=8, )
